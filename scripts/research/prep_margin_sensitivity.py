"""Quantify propagation of synthetic finish-line error into finished crown geometry."""
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from dental_ai.generation import prep_crown
from dental_ai.generation.learned import LearnedGenerator
from dental_ai.generation.mirror_template import MirrorTemplateGenerator
from dental_ai.margin.cervical import boundary_loops
from dental_ai.margin.detect import detect_margin, resample_loop
from dental_ai.meshdist import closest_point
from dental_ai.preferences import DesignPreferences

OUT = Path(os.environ.get("DENTAL_PREP_STUDY_OUT", "runs/prep_margin_sensitivity_20261122")).resolve()
SOURCE = Path(os.environ.get("DENTAL_SYNTHETIC_PAIRS", "runs/synthetic_prep_pairs_v1_20261121")).resolve()
PREFERENCES = DesignPreferences(occlusion_distance=0.0)
ARMS = ('truth', 'geometric_auto', 'occlusal_plus_030', 'apical_minus_030')
METRICS = ('input_line_mm','input_p95_mm','edge_mean_mm','edge_p95_mm','edge_max_mm',
           'intaglio_abs_mean_mm','intaglio_abs_p95_mm','prep_penetration_fraction',
           'thickness_p01_mm','thickness_min_mm','thin_fraction','occl_clearance_mm')
GENERATORS = {}


class FrozenOuter:
    def __init__(self, center):
        self.center = center
        self.result = None

    def generate(self, context, target):
        if self.result is None:
            if 'learned' not in GENERATORS:
                GENERATORS['mirror'] = MirrorTemplateGenerator(preferences=PREFERENCES)
                GENERATORS['learned'] = LearnedGenerator('models/generation/toothgen_open.pt',
                                                        fallback=GENERATORS['mirror'])
            try:
                result = GENERATORS['learned'].generate(context,target)
                if len(boundary_loops(result.crown)) != 1:
                    result = GENERATORS['mirror'].generate(context,target,center=self.center)
            except (LookupError, ValueError) as exc:
                raise RuntimeError('Fixed exterior generation failed') from exc
            if len(boundary_loops(result.crown)) != 1:
                raise RuntimeError('Could not obtain an open exterior with one boundary')
            self.result = result
        return self.result


def distances(a,b):
    a,b = resample_loop(a,400),resample_loop(b,400)
    return np.r_[cKDTree(a).query(b)[0],cKDTree(b).query(a)[0]]


def split_generated(mesh, source_die):
    """Recover the inner and outer surfaces from the assembly order of prep_crown."""
    border = len(mesh.vertices)-len(source_die.vertices)
    assert border > 0
    outer_faces = mesh.faces[(mesh.faces < border).all(axis=1)]
    inner_faces = mesh.faces[(mesh.faces >= border).all(axis=1)] - border
    assert len(inner_faces) == len(source_die.faces)
    assert np.array_equal(np.sort(inner_faces,axis=1),np.sort(source_die.faces,axis=1))
    return (trimesh.Trimesh(mesh.vertices[:border],outer_faces,process=False),
            trimesh.Trimesh(mesh.vertices[border:],source_die.faces,process=False))


def analyze_crown(result, source_die, reference_points, reference_normals, truth, input_dist):
    outer,inner=split_generated(result.mesh, source_die)
    rim = inner.vertices[boundary_loops(inner)[0]]
    d = distances(rim,truth)
    closest,abs_gap,_ = closest_point(inner,reference_points)
    signed=np.einsum('ij,ij->i',closest-reference_points,reference_normals)
    points=trimesh.sample.sample_surface(outer,5000,seed=2026)[0]
    fixed_band=cKDTree(resample_loop(truth,500)).query(points)[0]>1.5
    thickness=prep_crown._signed_thickness(points[fixed_band],inner)
    thickness=thickness[np.isfinite(thickness)]
    if len(thickness)<50:
        raise ValueError('Too few thickness samples away from the reference margin')
    return dict(watertight=bool(result.mesh.is_watertight), warnings=len(result.warnings),
                input_line_mm=float(input_dist.mean()),input_p95_mm=float(np.percentile(input_dist,95)),
                edge_mean_mm=float(d.mean()),edge_p95_mm=float(np.percentile(d,95)),edge_max_mm=float(d.max()),
                intaglio_abs_mean_mm=float(abs_gap.mean()),intaglio_abs_p95_mm=float(np.percentile(abs_gap,95)),
                prep_penetration_fraction=float(np.mean(signed<-.05)),thickness_p01_mm=float(np.percentile(thickness,1)),
                thickness_min_mm=float(thickness.min()),thin_fraction=float(np.mean(thickness<PREFERENCES.crown_thickness)),
                occl_clearance_mm=result.info.get('occlusal_clearance_mm'))


def evaluate_variant(record, tag, context):
    key=dict(case=record['id'],split=record['partition'],variant=tag)
    with np.load(SOURCE/'cases'/record['id']/tag/'variant.npz') as data:
        prepared=data['prep_vertices']
        truth=data['margin_points']
    mesh=trimesh.Trimesh(prepared,context['faces'],process=False)
    labels, up, fdi = context['labels'],np.array(context['up'],dtype=float),int(context['fdi'])
    up /= np.linalg.norm(up)
    antagonist=trimesh.Trimesh(context['opposing_vertices'],context['opposing_faces'],process=False)
    outer=FrozenOuter(resample_loop(truth,400).mean(axis=0))
    points_ref = closest_point(mesh,truth)[0]
    frame=prep_crown._frame(points_ref,up)
    ref_die=prep_crown.extract_die(mesh,points_ref,frame)
    sample, face_index=trimesh.sample.sample_surface(ref_die,5000,seed=2026)
    use=cKDTree(resample_loop(points_ref,500)).query(sample)[0]>1.0
    if use.sum()<100:
        raise ValueError('Insufficient truth preparation surface outside the margin transition')
    reference_points,reference_normals=sample[use],ref_die.face_normals[face_index[use]]
    lines=dict(truth=truth,occlusal_plus_030=truth+.30*up,apical_minus_030=truth-.30*up)
    rows={}
    try:
        lines['geometric_auto']=detect_margin(mesh,labels,fdi,up=up).points
    except (ValueError,IndexError,LookupError) as exc:
        rows['geometric_auto']={**key,'arm':'geometric_auto','error':f'{type(exc).__name__}: {exc}','stage':'detector'}
    generate_inner=prep_crown.make_intaglio
    for arm in ARMS:
        if arm in rows:
            continue
        captured=[]
        def capture(die,*args,**kwargs):
            captured.append(die)
            return generate_inner(die,*args,**kwargs)
        try:
            line=closest_point(mesh,np.asarray(lines[arm]))[0]
            input_dist=distances(line,points_ref)
            with patch.object(prep_crown,'make_intaglio',capture):
                result=prep_crown.generate_prep_crown(mesh,labels,record['jaw'],fdi,line,up,
                                    preferences=PREFERENCES,antagonist=antagonist,outer_generator=outer)
            if len(captured)!=1:
                raise RuntimeError('Could not identify the inner surface')
            metrics=analyze_crown(result,captured[0],reference_points,reference_normals,points_ref,input_dist)
            rows[arm]={**key,'arm':arm,**metrics}
        except (ValueError,IndexError,LookupError,RuntimeError) as exc:
            rows[arm]={**key,'arm':arm,'error':f'{type(exc).__name__}: {exc}','stage':'generation_or_measurement'}
    return [rows[a] for a in ARMS]


def work(record):
    dest=OUT/'cases'/f'{record["id"]}.json'
    if dest.is_file():
        return json.loads(dest.read_text())
    with np.load(SOURCE/'cases'/record['id']/'context.npz') as arrays:
        context={key:arrays[key] for key in ('faces','labels','up','fdi','opposing_vertices','opposing_faces')}
    rows=[]
    for v in record['variants']:
        tag=v['tag']
        try:
            rows.extend(evaluate_variant(record,tag,context))
        except (ValueError,IndexError,LookupError,RuntimeError) as exc:
            rows.extend(dict(case=record['id'],split=record['partition'],variant=tag,arm=arm,
                             stage='reference_geometry',error=f'{type(exc).__name__}: {exc}') for arm in ARMS)
    return rows


def write(path,payload):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload,indent=2,allow_nan=False))
    temp.replace(path)


def report_group(rows):
    arms={arm:{(r['case'],r['variant']):r for r in rows if r['arm']==arm} for arm in ARMS}
    accepted={arm:{k:r for k,r in group.items() if 'error' not in r and r['watertight']}
              for arm,group in arms.items()}
    base=accepted['truth']
    report={}
    rng=np.random.default_rng(20261122)
    for arm in ARMS:
        group=accepted[arm]
        shared=sorted(base.keys()&group.keys())
        stats={}
        for metric in METRICS:
            deltas=defaultdict(list)
            left,right=[],[]
            for key in shared:
                before,after=base[key].get(metric),group[key].get(metric)
                if before is None or after is None:
                    continue
                left.append(before)
                right.append(after)
                deltas[key[0]].append(after-before)
            if not deltas:
                continue
            by_patient=np.array([np.mean(v) for v in deltas.values()])
            boot=rng.choice(by_patient,(2000,len(by_patient)),replace=True).mean(axis=1)
            stats[metric]=dict(values=len(left),matched_truth_median=round(float(np.median(left)),5),
                               arm_median=round(float(np.median(right)),5),
                               patient_mean_paired_delta=round(float(by_patient.mean()),5),
                               patient_delta_ci=np.percentile(boot,[2.5,97.5]).round(5).tolist())
        report[arm]=dict(attempted=len(arms[arm]),watertight=len(group),error_stages=dict(Counter(r['stage'] for r in arms[arm].values() if 'error' in r)),
                         nonwatertight=sum('error' not in r and not r['watertight'] for r in arms[arm].values()),
                         paired_variants=len(shared),paired_patients=len({key[0] for key in shared}),metrics=stats)
    return report


def main():
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    source=json.loads((SOURCE/'manifest_private.json').read_text())
    jobs=[r for r in source['cases'] if 'error' not in r]
    assert len(jobs)==65 and sum(len(r['variants']) for r in jobs)==130
    jobs.sort(key=lambda r: r['partition']!='test')
    records=[]
    with ProcessPoolExecutor(max_workers=6,mp_context=get_context('spawn')) as pool:
        tasks={pool.submit(work,row):row for row in jobs}
        for i, task in enumerate(as_completed(tasks),1):
            data=task.result()
            record=tasks[task]
            write(OUT/'cases'/f'{record["id"]}.json',data)
            records.extend(data)
            if i==1 or i%4==0 or i==len(jobs):
                counts=Counter(record['arm'] for record in records if 'error' not in record and record['watertight'])
                print(json.dumps(dict(case_patients_finished=i,total_patients=len(jobs),
                                      watertight_by_arm=dict(counts))),flush=True)
    assert len(records)==130*len(ARMS)
    summary={fold:report_group([r for r in records if fold=='all' or r['split']==fold])
             for fold in ('test','dev','train','all')}
    write(OUT/'summary.json',dict(source_pairs=130,reference='simulated preparation and margin',
                                  test_cases=summary['test'],other_partitions={k:v for k,v in summary.items() if k!='test'}))
    print(json.dumps({'test_watertight':{k:v['watertight'] for k,v in summary['test'].items()},
                      'test_attempted':44,'source_pairs':130}),flush=True)
    if summary['test']['truth']['watertight'] < 12:
        raise SystemExit('Too few watertight truth crowns for interpretation')


if __name__=='__main__':
    main()
