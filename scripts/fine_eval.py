"""细小窝沟模型在网格上：线上 vs 线上 + 细小窝沟（两次采样），40 例。
量：三段尺度的幅度 / 和技师的位置相关（同 band_analysis）、窝沟深度、沟区 IoU / 相关、CD-L2、粗糙度。前 6 例存网格出图。
    FINE=runs/occl/fine.pt GAIN=1 python fine_eval.py 40 [offset]"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from trimesh.grouping import group_rows

sys.path.insert(0, str(Path.home() / "zyt/dental_3d_ai/ml"))
sys.path.insert(0, "/tmp")
from groove_lib import groove  # noqa: E402
from smooth_sweep_lib import rough  # noqa: E402
from dental_ai.eval_crownfusion import load_case, run_case, surface_points  # noqa: E402
from dental_ai.generation import occl_fine as of  # noqa: E402
from dental_ai.generation import occlusal as oc  # noqa: E402
from dental_ai.generation import voxel_crown as vc  # noqa: E402
from dental_ai.generation.learned import load_model  # noqa: E402
from dental_ai.generation.poisson import deepen_grooves, poisson_mesh  # noqa: E402

DATA = Path.home() / "zyt/dental_data/crownfusion/CrownFusion"
ROOT = Path.home() / "zyt/dental_3d_ai"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OFF = int(sys.argv[2]) if len(sys.argv) > 2 else 0
GAIN = float(os.environ.get("GAIN", 1.0))
vox = vc.load(ROOT / "models/generation/voxel_crown.pt")
shape, _ = load_model(ROOT / "models/generation/toothgen_ctx.pt", "cuda")
fm = of.load(ROOT / os.environ.get("FINE", "runs/occl/fine.pt"))
ids = sorted(json.load(open(DATA / "testing.json")), key=int)[OFF:OFF + N]
SIG = (0.055, 0.25, 0.6, 2.0)


def project(m, pm, blend=1.0, taubin=3):
    v = np.asarray(m.vertices).copy()
    es = m.edges_sorted
    rim = np.unique(es[group_rows(es, require_count=1)])
    w = np.clip(cKDTree(v[rim]).query(v)[0] / blend, 0, 1)
    cp, dist, _ = trimesh.proximity.closest_point(pm, v)
    w = w * (dist < 1.5)
    out = trimesh.Trimesh(v + w[:, None] * (cp - v), m.faces, process=False)
    if taubin:
        trimesh.smoothing.filter_taubin(out, lamb=0.5, nu=-0.53, iterations=taubin)
        out.vertices[rim] = v[rim]
    return out


def bands(h, m):
    ls = [oc.lowpass(h, m, s) for s in SIG]
    return ls[2] - ls[3], ls[1] - ls[2], ls[0] - ls[1]


rows = []
for cid in ids:
    d = DATA / "test" / str(cid)
    r = run_case(shape, d, 1.0, True, True, "cuda")
    if r is None:
        continue
    _, gen = r
    pts, props, prep, crown = load_case(d)
    pp = surface_points(prep, vc.N_PREP, int(cid) + 1)
    c = vc.crop_center(pp)
    bt = {"prep": torch.tensor(pp - c, dtype=torch.float32)[None],
          "ctx": torch.tensor(np.concatenate([pts[:, :3] - c, (props[:, 1:2] + 1)], 1), dtype=torch.float32)[None]}
    p, n = vc.predict_points(vox, vc.make_inputs(bt, "cuda"))[0]
    p, n = p.float().cpu().numpy() + c, n.float().cpu().numpy()
    v, f = vc.fit_mesh(np.asarray(gen.vertices), np.asarray(gen.faces), [], p, 150, 0.5)
    base = trimesh.Trimesh(*trimesh.remesh.subdivide(v, f), process=False)
    pm = poisson_mesh(p, n, 192, 1.0, 0.3)
    live = project(base, deepen_grooves(pm, 2.0))
    ant = pts[props[:, 1] == 1, :3].astype(float)
    meshes = {"线上": live}
    for s in (0, 1):
        meshes[f"细小窝沟{s + 1}"] = of.add_fine(live, ant, fm, seed=s, gain=GAIN)
    x4, (cx, cz, top) = oc.inputs(surface_points(pm, 300000, 1), ant)
    ht, mt = oc.coarse_map(surface_points(crown, 300000, 2), cx, cz, top)
    hm = {k: oc.coarse_map(surface_points(m, 300000, 3), cx, cz, top) for k, m in meshes.items()}
    allm = mt.copy()
    for h, m in hm.values():
        allm &= m
    vv = ndimage.binary_erosion(allm, iterations=int(0.8 / oc.CELL))
    if vv.sum() < 200:
        continue
    tb = bands(ht, mt)
    tg = oc.groove_depth_map(ht, mt)
    row = {"case": str(cid), "技师|groove": groove(surface_points(crown, 60000, 2))[0], "技师|rough": rough(crown)[1]}
    for i, b in enumerate(tb):
        row[f"技师|amp{i}"] = float(b[vv].std())
    b10 = surface_points(crown, 10000, 1)
    for k, m in meshes.items():
        h, mv = hm[k]
        for i, (b, t) in enumerate(zip(bands(h, mv), tb)):
            row[f"{k}|amp{i}"] = float(b[vv].std())
            row[f"{k}|cor{i}"] = float(np.corrcoef(b[vv], t[vv])[0, 1])
        gp = oc.groove_depth_map(h, mv)
        a_, b_ = gp[vv] > 0.08, tg[vv] > 0.08
        row[f"{k}|iou"] = float((a_ & b_).sum() / max((a_ | b_).sum(), 1))
        row[f"{k}|corr"] = float(np.corrcoef(gp[vv], tg[vv])[0, 1])
        a10 = surface_points(m, 10000, 1)
        row[f"{k}|cd_l2"] = float(np.sqrt((cKDTree(b10).query(a10)[0] ** 2).mean() + (cKDTree(a10).query(b10)[0] ** 2).mean()))
        row[f"{k}|groove"] = groove(surface_points(m, 60000, 2))[0]
        row[f"{k}|rough"] = rough(m)[1]
        if len(rows) < 6 and OFF == 0:
            m.export(f"/tmp/fg_{cid}_{k}.ply")
    if len(rows) < 6 and OFF == 0:
        crown.export(f"/tmp/fg_{cid}_技师.ply")
    rows.append(row)
    print(len(rows), {k: round(x, 3) for k, x in row.items() if "cor2" in k or "amp2" in k}, flush=True)
json.dump(rows, open(f"/tmp/fine_eval_rows_{OFF}.json", "w"))
print("FINE_EVAL_DONE")
