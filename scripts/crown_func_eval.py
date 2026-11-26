"""真实预备体测试集：补全指标（F-score、HD95/HD、法向一致性、邻接接触、咬合接触）+ 功能性后优化（邻接 / 咬合）。

网格 = 线上那套：形状模型网格贴到体素模型的点上（free rim）。生成的网格按 (模型, 病例) 缓存到 runs/func/mesh_<tag>/，
重跑后优化不用再生成。
    VOX=runs/voxel/vox.pt python crown_func_eval.py --limit 150 --tag v1
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path.home() / "zyt/dental_3d_ai/ml"))
from dental_ai.eval_crownfusion import load_case, margin_loop, occlusal_metrics, rim_points, run_case, surface_points  # noqa: E402
from dental_ai.generation import voxel_crown as vc  # noqa: E402
from dental_ai.generation.learned import antagonist_points, load_model  # noqa: E402

DATA = Path.home() / "zyt/dental_data/crownfusion/CrownFusion"
ROOT = Path.home() / "zyt/dental_3d_ai"
R = np.array([[0, 0, 1.0], [1, 0, 0], [0, 1, 0]])  # 标准 = 数据 @ R.T


def sample_n(mesh, n, seed):
    p, fi = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(p), np.asarray(mesh.face_normals)[fi]


def geo_metrics(m, gt):
    a, na = sample_n(m, 20000, 3)
    b, nb = sample_n(gt, 20000, 4)
    da, ia = cKDTree(b).query(a)
    db, ib = cKDTree(a).query(b)
    out = {}
    for t in (0.1, 0.2, 0.3, 0.5):
        P, Rc = (da < t).mean(), (db < t).mean()
        out[f"f@{t}"] = float(2 * P * Rc / max(P + Rc, 1e-9))
    out["hd95"] = float(max(np.percentile(da, 95), np.percentile(db, 95)))
    out["hd"] = float(max(da.max(), db.max()))
    out["nc"] = float((np.abs((na * nb[ia]).sum(1)).mean() + np.abs((nb * na[ib]).sum(1)).mean()) / 2)
    out["cd"] = float((da.mean() + db.mean()) / 2)
    return out


def outward(q, n):
    return -n if ((q - q.mean(0)) * n).sum(1).mean() < 0 else n


def contact_targets(pts, props):
    """邻牙（数据坐标，法向朝外）、对颌（标准坐标，antagonist_points：法向朝牙冠）。"""
    nb = {}
    for side, name in ((1, "m"), (-1, "d")):
        sel = (props[:, 1] == 0) & (props[:, 0] == side)
        if sel.sum() >= 200:
            q = pts[sel, :3].astype(float)
            nb[name] = (q, outward(q, pts[sel, 3:].astype(float)))
    ant = None
    antm = props[:, 1] == 1
    if antm.sum() >= 64:
        xyz, nrm = pts[:, :3].astype(float) @ R.T, pts[:, 3:].astype(float) @ R.T
        a6 = antagonist_points(xyz[antm], nrm[antm])
        ant = (a6[:, :3] @ R, a6[:, 3:] @ R)  # 回数据坐标
    return nb, ant


def signed(p, q, n, tree=None, far=3.0):
    tree = tree or cKDTree(q)
    d, i = tree.query(p)
    sd = ((p - q[i]) * n[i]).sum(1)
    return np.where(d < far, sd, np.inf)


def contact_metrics(m, nb, ant):
    p, _ = sample_n(m, 40000, 5)
    out = {}
    for k, (q, n) in nb.items():
        sd = signed(p, q, n)
        out[f"prox_{k}"] = float(sd.min())  # 负 = 穿进邻牙的深度；正 = 离邻牙最近的缝
    if ant is not None:
        sd = signed(p, *ant)
        out["occ_min"] = float(sd.min())
        out["occ_c01"] = float((sd < 0.1).mean() * m.area)  # 离对颌 0.1 mm 以内（含穿入）的面积 mm²
        out["occ_pen"] = float((sd < -0.05).mean() * m.area)
    return out


def postopt(m, nb, ant, prox_t=0.0, occ_t=0.0, pull_occ=False, iters=120, lap=2.0, keep=0.2):
    """底边不动，其余顶点位移；把邻接最小距离推到 prox_t（穿入的推出来、有缝的拉过去），对颌穿入的推出来（pull_occ 时也拉到 occ_t）。"""
    dev = "cuda"
    V0 = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=dev)
    F_ = torch.tensor(np.asarray(m.faces), dtype=torch.long, device=dev)
    rim = torch.tensor(np.unique(rim_points_idx(m)), dtype=torch.long, device=dev)
    free = torch.ones(len(V0), 1, device=dev)
    free[rim] = 0
    # 离底边越远越能动（底边附近只微动，颈缘不动）
    h = torch.cdist(V0, V0[rim]).min(1)[0]
    free = free * (h / 1.5).clamp(max=1)[:, None]
    e = torch.cat([F_[:, [0, 1]], F_[:, [1, 2]], F_[:, [2, 0]]]); e = torch.cat([e, e.flip(1)])
    deg = torch.zeros(len(V0), device=dev).index_add_(0, e[:, 0], torch.ones(len(e), device=dev))
    tg = []
    for q, n in nb.values():
        tg.append(("prox", torch.tensor(q, dtype=torch.float32, device=dev), torch.tensor(n, dtype=torch.float32, device=dev), prox_t, True))
    if ant is not None:
        tg.append(("occ", torch.tensor(ant[0], dtype=torch.float32, device=dev), torch.tensor(ant[1], dtype=torch.float32, device=dev), occ_t, pull_occ))
    D = torch.zeros_like(V0, requires_grad=True)
    opt = torch.optim.Adam([D], lr=0.01)
    for _ in range(iters):
        V = V0 + D * free
        loss = 0
        for kind, q, n, t, pull in tg:
            with torch.no_grad():
                dmin, idx = torch.cdist(V, q).min(1)
            near = dmin < 3.0
            if near.sum() == 0:
                continue
            sd = ((V[near] - q[idx[near]]) * n[idx[near]]).sum(1)
            loss = loss + torch.relu(t - sd).pow(2).sum() * 10  # 穿入（比目标近）的
            if pull:
                smin = -0.05 * torch.logsumexp(-sd / 0.05, 0)  # 软最小
                loss = loss + torch.relu(smin - t).pow(2) * 10
        Dm = D * free
        lp = Dm - torch.zeros_like(Dm).index_add_(0, e[:, 0], Dm[e[:, 1]]) / deg[:, None]
        loss = loss + lap * (lp ** 2).sum(1).sum() + keep * (Dm ** 2).sum(1).mean()
        if not torch.is_tensor(loss):
            break
        opt.zero_grad(); loss.backward(); opt.step()
    return trimesh.Trimesh((V0 + D * free).detach().cpu().numpy().astype(float), m.faces, process=False)


def rim_points_idx(m):
    from trimesh.grouping import group_rows
    es = m.edges_sorted
    return es[group_rows(es, require_count=1)].ravel()


def get_mesh(cid, vox, shape, cache):
    f = cache / f"{cid}.npz"
    if f.exists():
        z = np.load(f)
        return trimesh.Trimesh(z["v"], z["f"], process=False)
    d = DATA / "test" / str(cid)
    r = run_case(shape, d, 1.0, True, True, "cuda")
    if r is None:
        return None
    _, gen = r
    pts, props, prep, crown = load_case(d)
    pp = surface_points(prep, vc.N_PREP, int(cid) + 1)
    c = vc.crop_center(pp)
    bt = {"prep": torch.tensor(pp - c, dtype=torch.float32)[None],
          "ctx": torch.tensor(np.concatenate([pts[:, :3] - c, (props[:, 1:2] + 1)], 1), dtype=torch.float32)[None]}
    p, _ = vc.predict_points(vox, vc.make_inputs(bt, "cuda"))[0]
    p = p.cpu().numpy() + c
    v, fa = vc.fit_mesh(np.asarray(gen.vertices), np.asarray(gen.faces), [], p, 150, 0.5, 1.0)
    np.savez(f, v=v.astype(np.float32), f=fa.astype(np.int32))
    return trimesh.Trimesh(v, fa, process=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--variants", default="prox0,prox0_occ,prox0_occpull")
    a = ap.parse_args()
    out = ROOT / "runs/func"; cache = out / f"mesh_{a.tag}"; cache.mkdir(parents=True, exist_ok=True)
    vox = vc.load(Path(os.environ["VOX"]))
    shape, _ = load_model(ROOT / "models/generation/toothgen_ctx.pt", "cuda")
    ids = sorted(json.load(open(DATA / "testing.json")), key=int)[: a.limit]
    V = {"prox0": dict(prox_t=0.0, occ_t=-1e3), "prox0_occ": dict(prox_t=0.0, occ_t=0.0),
         "prox0_occpull": dict(prox_t=0.0, occ_t=0.0, pull_occ=True)}
    rows = []
    t0 = time.time()
    for k, cid in enumerate(ids):
        try:
            m = get_mesh(cid, vox, shape, cache)
            if m is None:
                continue
            pts, props, prep, crown = load_case(DATA / "test" / str(cid))
            nb, ant = contact_targets(pts, props)
            row = {"case": str(cid)}
            for name, mm in [("gt", crown), ("gen", m)]:
                if name != "gt":
                    row.update({f"{name}_{kk}": v for kk, v in geo_metrics(mm, crown).items()})
                row.update({f"{name}_{kk}": v for kk, v in contact_metrics(mm, nb, ant).items()})
            for vn in a.variants.split(","):
                kw = dict(V[vn])
                if kw.get("occ_t", 0) < -100:
                    kw["occ_t"] = 0.0; ant_use = None
                else:
                    ant_use = ant
                mo = postopt(m, nb, ant_use, **kw)
                row.update({f"{vn}_{kk}": v for kk, v in geo_metrics(mo, crown).items()})
                row.update({f"{vn}_{kk}": v for kk, v in contact_metrics(mo, nb, ant).items()})
                if k < 8:
                    mo.export(cache / f"{cid}_{vn}.ply")
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            print("skip", cid, type(exc).__name__, exc, flush=True)
        if k % 25 == 0:
            print(k, round(time.time() - t0), flush=True)
    json.dump(rows, open(out / f"rows_{a.tag}.json", "w"))
    keys = sorted({kk for r in rows for kk in r if kk != "case"})
    s = {"n": len(rows)}
    for kk in keys:
        v = np.array([r[kk] for r in rows if kk in r and np.isfinite(r[kk])])
        s[kk] = {"median": round(float(np.median(v)), 3), "mean": round(float(v.mean()), 3), "n": len(v)}
    json.dump(s, open(out / f"summary_{a.tag}.json", "w"), indent=1)
    print("FUNC_DONE", json.dumps(s))


if __name__ == "__main__":
    main()
