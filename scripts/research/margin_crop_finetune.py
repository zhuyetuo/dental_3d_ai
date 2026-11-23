"""Matched-budget real-die margin finetuning with geometry-only crop augmentation."""
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from dental_ai.margin import learned

OUT = Path(os.environ.get("DENTAL_MARGIN_CROP_OUT", "runs/margin_crop_finetune_20261122")).resolve()
BASE = Path('models/margin/margin.pt')
SOURCES = {'incisor': (Path('data/prep_margin/training Data'), Path('data/prep_margin/test Data')),
           'public_die': (Path('data/margin_public/split/train'), Path('data/margin_public/split/test'))}
SEED = 2211
EPOCHS, STEPS, BATCH = 8, 28, 3
DEVICE = 'cuda'


def patient(path):
    return path.name.split('-', 1)[0]


def crop(mesh, target, line, level):
    """Cut fixed tails of the longest mesh axis, then require visible reference coverage."""
    vertices = np.asarray(mesh.vertices)
    center = vertices.mean(axis=0)
    _, basis = np.linalg.eigh(np.cov((vertices - center).T))
    axis = basis[:, -1]
    projection = (vertices - center) @ axis
    q = np.quantile(projection, [level, 1 - level])
    keep = (projection >= q[0]) & (projection <= q[1])
    faces = np.asarray(mesh.faces)
    selected = np.flatnonzero(keep[faces].all(axis=1))
    if len(selected) < 750:
        return None
    clipped = mesh.submesh([selected], append=True, repair=False)
    cv = np.asarray(clipped.vertices)
    if len(cv) < 800 or len(cv) > .90 * len(vertices):
        return None
    # Require the crop to preserve the annotated line as a controlled partial-input test.
    if np.mean(cKDTree(cv).query(line)[0] <= .20) < .975:
        return None
    source = cKDTree(vertices).query(cv)[1]
    return clipped, learned.features_of(clipped), target[source], line


def load(path):
    full = learned.case_arrays(path)
    clipped = [item for trim in (.12, .20) if (item := crop(full[0], full[2], full[3], trim)) is not None]
    return path, full, clipped


def evaluate(model, dataset, cropped):
    rows = []
    model.eval()
    for path, full, variants in dataset:
        for i, payload in enumerate(variants if cropped else [full]):
            mesh, feats, _, line = payload
            try:
                prediction, _ = learned.detect(model, mesh, DEVICE, feats)
                a = cKDTree(line).query(prediction)[0]
                b = cKDTree(prediction).query(line)[0]
                err = float((a.mean() + b.mean()) / 2)
                if not np.isfinite(err):
                    raise ValueError('Nonfinite line error')
                metrics = dict(mean_mm=err, p95_mm=float(np.percentile(np.r_[a,b],95)), failed=False)
            except (ValueError, IndexError):
                metrics = dict(mean_mm=5., p95_mm=None, failed=True)
            rows.append(dict(patient=patient(path), case=path.stem, variant=i, **metrics))
    return rows


def metric(rows):
    if not rows:
        raise ValueError('No valid evaluation inputs')
    good = [r for r in rows if not r['failed']]
    return dict(samples=len(rows), patients=len({r['patient'] for r in rows}),
                mean_with_failure_penalty_mm=round(float(np.mean([r['mean_mm'] for r in rows])), 5),
                median_mm=round(float(np.median([r['mean_mm'] for r in good])), 5) if good else None,
                median_p95_mm=round(float(np.median([r['p95_mm'] for r in good])), 5) if good else None,
                failures=len(rows)-len(good))


def model_at(path):
    model = learned._model().to(DEVICE)
    state = torch.load(path, map_location='cpu', weights_only=True)['state']
    model.load_state_dict(state)
    return model


def fit(name, train, dev, starting_metric):
    model = model_at(BASE)
    torch.manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=6e-5, total_steps=EPOCHS*STEPS, pct_start=.12)
    draw = np.random.default_rng(SEED)
    crop_draw = np.random.default_rng(SEED + 11)
    best, best_path = starting_metric, BASE
    ckpt = OUT/f'{name}.pt'
    for epoch in range(1,EPOCHS+1):
        model.train()
        running = torch.zeros((), device=DEVICE)
        for _ in range(STEPS):
            x, y = [], []
            for i in draw.choice(len(train),BATCH,replace=False):
                _, full, variants = train[i]
                plan = crop_draw.random()
                choice = crop_draw.integers(max(len(variants),1))
                _, feats, target, _ = variants[choice] if name == 'crop_augmented' and variants and plan < .65 else full
                idx = draw.choice(len(feats),learned.N_POINTS,replace=len(feats)<learned.N_POINTS)
                transform = learned._random_rotation(draw,draw.random()<.5)
                points = learned._prepare(feats[idx],transform,draw.uniform(.9,1.1),
                                          draw.normal(0,.05,3).astype(np.float32))
                x.append(points.T)
                y.append(target[idx])
            feat = torch.as_tensor(np.stack(x),device=DEVICE)
            target = torch.as_tensor(np.stack(y),device=DEVICE)
            pred = model(feat)[:,0]
            w = 1+4*torch.exp(-((target.abs()*learned.CLIP/.5)**2))
            loss = (w*torch.nn.functional.smooth_l1_loss(pred,target,beta=.05,reduction='none')).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()
            running += loss.detach()
        if epoch % 2 == 0:
            full = metric(evaluate(model,dev,False))
            cropped = metric(evaluate(model,dev,True))
            value = .5*(full['mean_with_failure_penalty_mm']+cropped['mean_with_failure_penalty_mm'])
            print(json.dumps(dict(arm=name,epoch=epoch,loss=running.item()/STEPS,
                                  dev_full_mm=full['mean_with_failure_penalty_mm'],
                                  dev_cropped_mm=cropped['mean_with_failure_penalty_mm'])),flush=True)
            state = dict(state=model.state_dict(),epoch=epoch,development_score_mm=value)
            torch.save(state,OUT/f'{name}.last.pt')
            if value < best:
                torch.save(state,ckpt)
                best, best_path = value, ckpt
    return best_path


def ci(base, candidate):
    differences=defaultdict(list)
    for a,b in zip(base,candidate,strict=True):
        assert (a['case'],a['variant']) == (b['case'],b['variant'])
        differences[a['patient']].append(b['mean_mm']-a['mean_mm'])
    by_patient=np.array([np.mean(vals) for vals in differences.values()])
    resample=np.random.default_rng(SEED).choice(by_patient,(3000,len(by_patient)),replace=True)
    return np.percentile(resample.mean(axis=1),[2.5,97.5]).round(5).tolist()


def main():
    assert torch.cuda.is_available()
    OUT.mkdir(parents=True, exist_ok=True)
    train_files={name: learned.case_files(paths[0]) for name,paths in SOURCES.items()}
    test_files={name: learned.case_files(paths[1]) for name,paths in SOURCES.items()}
    train_pat={patient(f) for files in train_files.values() for f in files}
    test_pat={patient(f) for files in test_files.values() for f in files}
    assert not train_pat & test_pat
    rng=np.random.default_rng(SEED)
    dev_pat=set()
    for cases in train_files.values():
        people=sorted({patient(p) for p in cases})
        dev_pat.update(rng.choice(people,max(3,round(.14*len(people))),replace=False))
    train,dev=[],[]
    errors=[]
    for paths in train_files.values():
        for path in paths:
            try:
                data=load(path)
            except (ValueError, IndexError) as exc:
                errors.append(f'{type(exc).__name__}: {exc}')
                continue
            (dev if patient(path) in dev_pat else train).append(data)
    counts=dict(train_cases=len(train),development_cases=len(dev),train_cropped=sum(len(r[2]) for r in train),
                development_cropped=sum(len(r[2]) for r in dev),skipped_training_cases=len(errors))
    print(json.dumps(counts),flush=True)
    assert len(train)>=65 and len(dev)>=8 and counts['train_cropped']>=30 and counts['development_cropped']>=5
    assert not ({patient(row[0]) for row in train} & {patient(row[0]) for row in dev})
    initial=model_at(BASE)
    full=metric(evaluate(initial,dev,False))['mean_with_failure_penalty_mm']
    clipped=metric(evaluate(initial,dev,True))['mean_with_failure_penalty_mm']
    start=(full+clipped)/2
    print(json.dumps(dict(initial_dev_full_mm=full,initial_dev_cropped_mm=clipped)),flush=True)
    del initial
    models={'initial':BASE,'standard':fit('standard',train,dev,start),'crop_augmented':fit('crop_augmented',train,dev,start)}
    tests={key:[load(path) for path in paths] for key,paths in test_files.items()}
    results={}
    for name,path in models.items():
        model=model_at(path)
        results[name]={key:{view:evaluate(model,data,view=='cropped') for view in ('complete','cropped')}
                       for key,data in tests.items()}
    matrix={key:{name:{view:metric(r) for view,r in result[key].items()}
                 for name,result in results.items()} for key in tests}
    intervals={key:ci(results['standard'][key]['cropped'],results['crop_augmented'][key]['cropped']) for key in tests}
    safe=all(matrix[k]['crop_augmented']['complete']['mean_with_failure_penalty_mm'] <=
             matrix[k]['initial']['complete']['mean_with_failure_penalty_mm']+.02
             and matrix[k]['crop_augmented']['complete']['failures'] <= matrix[k]['initial']['complete']['failures']
             and matrix[k]['crop_augmented']['cropped']['failures'] <= matrix[k]['standard']['cropped']['failures']
             for k in tests)
    supported=safe and any(v[1]<0 for v in intervals.values()) and all(v[1]<=.03 for v in intervals.values())
    report=dict(decision='replicate_on_real_partial_scans' if supported else 'keep_existing_die_model',
                real_die_margin_training=dict(training=counts,summary=matrix,crop_vs_standard_patient_ci_mm=intervals,
                                              selected_checkpoints={k:p.name for k,p in models.items()}))
    (OUT/'scores_private.json').write_text(json.dumps(results,indent=2))
    tmp=OUT/'summary.tmp'
    tmp.write_text(json.dumps(report,indent=2))
    tmp.replace(OUT/'summary.json')
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__':
    main()
