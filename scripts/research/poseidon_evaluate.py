"""Full-mesh paired evaluation with one shared postprocessing path for all models."""
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

import numpy as np

from dental_ai.evaluation import find_scans, load_instances, segmentation_metrics
from dental_ai.io import load_labels, load_mesh
from dental_ai.orientation import normalize_orientation
from dental_ai.segmentation.baseline import fdi_to_class
from dental_ai.segmentation.deep import DeepSegmenter
from dental_ai.segmentation.numbering import correct_numbering
from dental_ai.segmentation.postprocess import merge_split_teeth, remove_small_islands

from poseidon_finetune import OUT, POSEIDON, SEED, measures, slots, val_stats

MODEL_NAMES = ('initial','uniform','hard_weighted')
GROUPS = ('complex','simple','fair')
_MODELS = {}


def get_models():
    if not _MODELS:
        paths = json.loads((OUT/'selected.json').read_text())
        _MODELS.update({key:DeepSegmenter(paths[key],device='cuda') for key in MODEL_NAMES})
    return _MODELS


def process(task):
    group, scan = task
    mesh = load_mesh(scan.mesh_path)
    gt,_ = load_labels(scan.gt_path)
    instances = load_instances(scan.gt_path)
    canonical = normalize_orientation(mesh,scan.jaw).apply(mesh)
    jaw = scan.jaw
    order = slots(jaw)
    y = fdi_to_class(gt,jaw)
    models = {}
    for name,model in get_models().items():
        try:
            pred = model.predict(canonical,jaw)
            pred = remove_small_islands(mesh,correct_numbering(mesh,pred,jaw,canonical_vertices=canonical.vertices).labels)
            pred = merge_split_teeth(mesh,pred,jaw,points=np.asarray(canonical.vertices))[0]
            score, _ = segmentation_metrics(gt,pred,mesh.vertices,instances)
            slot_result = measures(y,fdi_to_class(pred,jaw),order)
            substitutions = 0
            for fdi in np.unique(gt):
                if fdi == 0:
                    continue
                values,counts = np.unique(pred[gt==fdi],return_counts=True)
                idx = int(np.argmax(counts))
                substitutions += int(values[idx] not in (0,fdi) and counts[idx]>.5*counts.sum())
            models[name] = dict(t_miou=score['t_miou'],tir=score['tir'],score=score['score'],
                                fdi_substitutions=substitutions,**slot_result)
        except (ValueError,IndexError,RuntimeError) as exc:
            models[name] = dict(error=f'{type(exc).__name__}: {exc}'[:240])
    return dict(group=group,patient=scan.case,scan=scan.name,models=models)


def confidence(rows, candidate, control, metric):
    patients=defaultdict(list)
    for r in rows:
        a,b=r['models'][candidate],r['models'][control]
        patients[r['patient']].append(a[metric]-b[metric])
    values=np.array([np.mean(v) for v in patients.values()])
    boot=np.random.default_rng(SEED).choice(values,(3000,len(values)),replace=True).mean(axis=1)
    return np.percentile(boot,[2.5,97.5]).round(5).tolist()


def summarize(data):
    results={}
    for group in GROUPS:
        rows=[r for r in data if r['group']==group]
        paired=[r for r in rows if all('error' not in r['models'][name] for name in MODEL_NAMES)]
        if not paired:
            raise ValueError(f'No matched output for {group}')
        scores={}
        for model in MODEL_NAMES:
            subset=[r['models'][model] for r in paired]
            metrics={k:round(float(np.mean([r[k] for r in subset])),5) for k in ('t_miou','tir','score')}
            slot=val_stats(subset)
            del slot['point_sample_tmIoU']
            scores[model]=dict(failures=sum('error' in row['models'][model] for row in rows),
                               **metrics,**slot,fdi_substitutions=sum(r['fdi_substitutions'] for r in subset))
        results[group]=dict(expected_scans=len(rows),paired_scans=len(paired),paired_patients=len({r['patient'] for r in paired}),models=scores,
                             hard_vs_uniform_tmIoU_ci=confidence(paired,'hard_weighted','uniform','t_miou'),
                             hard_vs_baseline_tmIoU_ci=confidence(paired,'hard_weighted','initial','t_miou'))
    return results


def main():
    from pathlib import Path
    folders={'complex':POSEIDON/'test-complex','simple':POSEIDON/'test-simple','fair':Path('data/fair_test')}
    scans={name:find_scans(path,path) for name,path in folders.items()}
    assert len(scans['complex'])==len(scans['simple'])==20 and len(scans['fair'])>=100
    rows=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as pool:
        pending=[pool.submit(process,(name,scan)) for name in GROUPS for scan in scans[name]]
        for i,task in enumerate(as_completed(pending),1):
            rows.append(task.result())
            if i%10==0 or i==len(pending):
                print(f'{i}/{len(pending)} full meshes scored',flush=True)
    rows.sort(key=lambda r:(r['group'],r['patient'],r['scan']))
    results=summarize(rows)
    bound=results['complex']
    hard=bound['models']['hard_weighted']
    primary=all(hard['internal_missing_detected'] >= bound['models'][k]['internal_missing_detected'] and
                hard['fdi_substitutions'] <= bound['models'][k]['fdi_substitutions'] and
                hard['false_internal_missing'] <= bound['models'][k]['false_internal_missing']
                for k in ('initial','uniform'))
    benefit=(hard['internal_missing_detected'] > bound['models']['initial']['internal_missing_detected'] or
             hard['fdi_substitutions'] < bound['models']['initial']['fdi_substitutions'])
    robustness=all(g['models']['hard_weighted']['t_miou'] >= g['models']['initial']['t_miou']-.005
                   and g['models']['hard_weighted']['failures'] <= g['models']['initial']['failures']
                   for g in results.values())
    sufficient=all(g['paired_scans']==g['expected_scans'] for g in results.values())
    report=dict(decision='hard_sampling_warrants_replication' if primary and benefit and robustness and sufficient else 'keep_existing_segmenter',
                paired_segmentation_results=results,selected_checkpoints=json.loads((OUT/'selected.json').read_text()))
    (OUT/'full_scores_private.json').write_text(json.dumps(rows,indent=2))
    tmp=OUT/'summary.tmp'
    tmp.write_text(json.dumps(report,indent=2))
    tmp.replace(OUT/'summary.json')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
