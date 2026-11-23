"""Compare neutral and difficult-case sampling from one fixed segmentation checkpoint."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import cross_entropy

from dental_ai.evaluation import find_scans
from dental_ai.fdi import arch_order
from dental_ai.segmentation.deep import N_CLASSES, build_model
from dental_ai.segmentation.baseline import fdi_to_class

OUT = Path(os.environ.get("DENTAL_POSEIDON_OUT", "runs/poseidon_hard_finetune_20261122")).resolve()
CACHE = Path('data/deep_cache')
POSEIDON = Path('data/poseidon3d')
BASE = Path('models/segmentation/deep.pt')
SEED = 221122
EPOCHS, STEPS, BATCH = 6, 72, 4
DEVICE = 'cuda'


def scan_files(directory):
    return find_scans(directory, directory)


def arrays(paths, points, rng):
    p = Path(paths)
    with np.load(p) as source:
        idx = rng.choice(len(source['y']), points, replace=len(source['y']) < points)
        return (source['x'][idx].astype(np.float32),source['y'][idx].astype(np.int64))


def slots(jaw):
    return fdi_to_class(np.asarray(arch_order(jaw)),jaw)


def slot_sets(labels, order):
    counts = np.bincount(labels, minlength=N_CLASSES)
    return {int(c) for c in order if counts[c] >= max(10, len(labels)*.002)}


def holes(present, order):
    hits = [i for i, c in enumerate(order) if c in present]
    return set(order[hits[0]:hits[-1]+1]) - present if len(hits)>=2 else set()


def measures(y, p, order):
    gt, pred = slot_sets(y,order), slot_sets(p,order)
    true_holes, pred_holes = holes(gt,order), holes(pred,order)
    labels = [c for c in np.unique(y) if c]
    ious = [float(np.sum((y==c)&(p==c))/np.sum((y==c)|(p==c))) for c in labels]
    return dict(iou=float(np.mean(ious)) if ious else 0., missing=len(true_holes),
                missing_correct=len(true_holes&pred_holes), extra_holes=len(pred_holes-true_holes),
                wrong_presence=len(gt^pred), exact_slots=gt==pred)


def new_model(initial):
    model = build_model(arch=initial['arch']).to(DEVICE)
    model.load_state_dict(initial['state_dict'])
    return model


def logits_batch(model, payload):
    return model(torch.from_numpy(np.stack([row[0].T for row in payload])).to(DEVICE))


def val_rows(model, entries):
    model.eval()
    rows = []
    with torch.inference_mode():
        for a in range(0, len(entries), BATCH):
            group = entries[a:a+BATCH]
            pred = logits_batch(model, [row[0] for row in group]).argmax(dim=1).cpu().numpy()
            rows.extend(measures(x[1], p, order) for (x,order),p in zip(group,pred,strict=True))
    return rows


def val_stats(rows):
    return dict(point_sample_tmIoU=round(float(np.mean([r['iou'] for r in rows])),5),
                exact_slot_scans=sum(r['exact_slots'] for r in rows),
                annotated_internal_missing=sum(r['missing'] for r in rows),
                internal_missing_detected=sum(r['missing_correct'] for r in rows),
                false_internal_missing=sum(r['extra_holes'] for r in rows),
                slot_errors=sum(r['wrong_presence'] for r in rows))


def selection(rows):
    return (float(np.mean([r['iou'] for r in rows])) -
            .015 * float(np.mean([r['wrong_presence'] for r in rows])))


def main():
    assert torch.cuda.is_available()
    OUT.mkdir(parents=True, exist_ok=True)
    initial = torch.load(BASE, map_location='cpu', weights_only=True)
    stats = initial['stats']
    mean,std = np.array(stats['mean'],np.float32),np.array(stats['std'],np.float32)
    points = min(int(initial['n_points']), 6000)
    poseidon = scan_files(POSEIDON/'train')
    validation = scan_files(OUT/'validation_meshes')
    heldout = scan_files(POSEIDON/'test-complex') + scan_files(POSEIDON/'test-simple')
    assert len(poseidon)==150 and len(validation)==10 and len(heldout)==40
    patient_sets = [{p.case for p in group} for group in (poseidon,validation,heldout)]
    assert len(set.union(*patient_sets)) == sum(map(len,patient_sets))
    cached_poseidon = [CACHE/f'{s.name}.npz' for s in poseidon]
    assert all(p.is_file() for p in cached_poseidon)
    blocked = {name.rsplit('_',1)[0] for name in Path('data/fair_test.txt').read_text().splitlines() if name}
    blocked |= {s.case for s in scan_files(Path('data/fair_test'))}
    blocked |= {s.case for s in scan_files(Path('data/challenge_test100'))}
    blocked |= {s.case for s in scan_files(Path('data/teeth3ds_test100'))}
    replay = [p for p in sorted(CACHE.glob('*.npz')) if '__' not in p.stem
              and p.stem.rsplit('_',1)[0] not in blocked
              and p.stem.rsplit('_',1)[0] not in set.union(*patient_sets)]
    assert len(replay)>=150
    val_files = [OUT/'validation_cache'/f'{s.name}.npz' for s in validation]
    assert all(p.is_file() for p in val_files)
    def fixed(paths, scans, seed):
        data = []
        for i,(path,scan) in enumerate(zip(paths,scans,strict=True)):
            x,y = arrays(path,points,np.random.default_rng(seed+i))
            data.append((((x-mean)/std,y),slots(scan.jaw)))
        return data
    dev = fixed(val_files,validation,SEED+1000)
    base = new_model(initial)
    base_val = val_rows(base,dev)
    hard = val_rows(base,fixed(cached_poseidon,poseidon,SEED+2000))
    difficulty = sorted(range(len(hard)), key=lambda i: (2*(hard[i]['missing']-hard[i]['missing_correct'])+
                  hard[i]['wrong_presence']+.25*hard[i]['missing']+2*(1-hard[i]['iou']),-i), reverse=True)
    hard_ids = set(difficulty[:len(hard)//3])
    weights = np.array([3. if i in hard_ids else 1. for i in range(len(hard))])
    print(json.dumps(dict(poseidon_train=len(poseidon), poseidon_development=len(validation),
                          teeth3ds_replay=len(replay), difficult_train_scans=len(hard_ids),
                          optimizer_steps_per_arm=EPOCHS*STEPS,train_points=points,
                          baseline_development=val_stats(base_val))),flush=True)
    del base

    train_paths = [*cached_poseidon,*replay]
    counts = np.zeros(N_CLASSES)
    for path in train_paths:
        with np.load(path) as x:
            counts += np.bincount(x['y'],minlength=N_CLASSES)
    weight = np.sqrt(counts.sum()/np.maximum(counts,1))
    class_weights = torch.tensor(weight/weight.mean(),dtype=torch.float32,device=DEVICE)
    model_paths = {'initial':str(BASE)}
    for arm in ('uniform','hard_weighted'):
        torch.manual_seed(SEED)
        model = new_model(initial)
        opt = torch.optim.AdamW(model.parameters(),lr=4e-5,weight_decay=1e-4)
        decay = torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=4e-5,total_steps=EPOCHS*STEPS,pct_start=.15)
        pose_draw = np.random.default_rng(SEED+1)
        other_draw = np.random.default_rng(SEED+2)
        aug = np.random.default_rng(SEED+3)
        best,best_path = -np.inf,BASE
        for epoch in range(1,EPOCHS+1):
            model.train()
            running=torch.zeros((),device=DEVICE)
            for _ in range(STEPS):
                case_ids=pose_draw.choice(len(cached_poseidon),2,replace=False,p=(weights/weights.sum() if arm=='hard_weighted' else None))
                replay_ids=other_draw.choice(len(replay),2,replace=False)
                paths=[cached_poseidon[i] for i in case_ids]+[replay[i] for i in replay_ids]
                payload = []
                for path in paths:
                    x,y=arrays(path,points,aug)
                    x=((x-mean)/std)*aug.uniform(.95,1.05)+aug.normal(0,.01,x.shape).astype(np.float32)
                    payload.append((x,y))
                pred=logits_batch(model,payload)
                truth=torch.from_numpy(np.stack([y for _,y in payload])).to(DEVICE)
                loss=cross_entropy(pred,truth,weight=class_weights)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                decay.step()
                running+=loss.detach()
            rows=val_rows(model,dev)
            value=selection(rows)
            report=dict(arm=arm,epoch=epoch,loss=round(float(running.item()/STEPS),6),**val_stats(rows))
            print(json.dumps(report),flush=True)
            record=dict(state_dict=model.state_dict(),stats=stats,n_points=initial['n_points'],
                        arch=initial['arch'],development=report)
            torch.save(record,OUT/f'{arm}.last.pt')
            if value > best:
                best,best_path=value,OUT/f'{arm}.pt'
                torch.save(record,best_path)
        model_paths[arm]=str(best_path)
    (OUT/'selected.json').write_text(json.dumps(model_paths,indent=2))


if __name__ == '__main__':
    main()
