"""咬合面模型在网格上的效果（真实预备体测试集）：线上（泊松 + 窝沟加深 2 倍）vs 咬合面模型（泊松表面叠残差再投影）vs 两者叠加。
指标：窝沟深度（groove_p90）、窝沟位置（和技师牙冠的沟图：沟区 IoU、沟深相关）、牙尖 / 窝、CD-L2、粗糙度。
    OCCL=runs/occl/occl.pt python occl_eval.py 40"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path.home() / "zyt/dental_3d_ai/ml"))
sys.path.insert(0, "/tmp")
from groove_lib import groove  # noqa: E402
from smooth_sweep_lib import marks, rough  # noqa: E402
from dental_ai.eval_crownfusion import load_case, run_case, surface_points  # noqa: E402
from dental_ai.generation import occlusal as oc  # noqa: E402
from dental_ai.generation import voxel_crown as vc  # noqa: E402
from dental_ai.generation.learned import load_model  # noqa: E402
from dental_ai.generation.poisson import deepen_grooves, poisson_mesh  # noqa: E402
from trimesh.grouping import group_rows  # noqa: E402

DATA = Path.home() / "zyt/dental_data/crownfusion/CrownFusion"
ROOT = Path.home() / "zyt/dental_3d_ai"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
vox = vc.load(ROOT / "models/generation/voxel_crown.pt")
shape, _ = load_model(ROOT / "models/generation/toothgen_ctx.pt", "cuda")
occl = oc.load(ROOT / os.environ.get("OCCL", "runs/occl/occl.pt"))
ids = sorted(json.load(open(DATA / "testing.json")), key=int)[:N]


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


def groove_pos(m, tech_map, valid, frame):
    """生成牙冠和技师牙冠在同一高度图坐标里的沟图：沟区（沟深 > 0.08 mm）IoU、沟深相关系数。"""
    cx, cz, top = frame
    h, mv = oc.coarse_map(surface_points(m, 300000, 3), cx, cz, top)
    v = valid & mv
    from scipy import ndimage

    v = ndimage.binary_erosion(v, iterations=int(0.8 / oc.CELL))  # 只看内圈（离轮廓 0.8 mm 以上）
    if v.sum() < 100:
        return np.nan, np.nan
    gp, gt = oc.groove_depth_map(h, mv), tech_map
    a, b = gp[v] > 0.08, gt[v] > 0.08
    iou = (a & b).sum() / max((a | b).sum(), 1)
    corr = np.corrcoef(gp[v], gt[v])[0, 1]
    return float(iou), float(corr)


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
    try:
        g = marks(crown)
    except Exception:  # noqa: BLE001
        continue
    v, f = vc.fit_mesh(np.asarray(gen.vertices), np.asarray(gen.faces), [], p, 150, 0.5)
    sv, sf = trimesh.remesh.subdivide(v, f)
    base = trimesh.Trimesh(sv, sf, process=False)
    pm = poisson_mesh(p, n, 192, 1.0, 0.3)
    ant = pts[props[:, 1] == 1, :3].astype(float)
    coarse = surface_points(pm, 300000, 1)
    resid, valid, frame = oc.predict(occl, coarse, ant)
    pm_occ = oc.apply_residual(pm, resid, valid, frame)
    meshes = {"不加深": project(base, pm), "线上(加深2)": project(base, deepen_grooves(pm, 2.0)),
              "咬合面模型": project(base, pm_occ), "咬合面模型+加深1": project(base, deepen_grooves(pm_occ, 1.0))}
    cx, cz, top = frame
    ht, mt = oc.coarse_map(surface_points(crown, 300000, 2), cx, cz, top)
    tech_g = oc.groove_depth_map(ht, mt)
    row = {"case": str(cid)}
    row["技师|groove"] = groove(surface_points(crown, 60000, 2))[0]
    row["技师|rough"] = rough(crown)[1]
    b10 = surface_points(crown, 10000, 1)
    for name, m in meshes.items():
        a10 = surface_points(m, 10000, 1)
        row[f"{name}|cd_l2"] = float(np.sqrt((cKDTree(b10).query(a10)[0] ** 2).mean() + (cKDTree(a10).query(b10)[0] ** 2).mean()))
        row[f"{name}|groove"] = groove(surface_points(m, 60000, 2))[0]
        row[f"{name}|rough"] = rough(m)[1]
        row[f"{name}|iou"], row[f"{name}|corr"] = groove_pos(m, tech_g, mt, frame)
        try:
            k = marks(m)
            row[f"{name}|cusp"] = float(cKDTree(k["cusps"]).query(g["cusps"])[0].mean())
            row[f"{name}|fossa"] = float(np.linalg.norm(k["fossa"] - g["fossa"]))
        except Exception:  # noqa: BLE001
            pass
        if len(rows) < 6:
            m.export(f"/tmp/oc_{cid}_{name}.ply")
    if len(rows) < 6:
        crown.export(f"/tmp/oc_{cid}_技师.ply")
    rows.append(row)
    print(len(rows), {k: round(x, 3) for k, x in row.items() if "iou" in k or "groove" in k}, flush=True)
json.dump(rows, open("/tmp/occl_eval_rows.json", "w"))
print("n", len(rows))
for name in ["技师"] + list(meshes):
    s = []
    for k in ("groove", "iou", "corr", "cusp", "fossa", "cd_l2", "rough"):
        a = np.array([r.get(f"{name}|{k}", np.nan) for r in rows], float)
        a = a[np.isfinite(a)]
        if len(a):
            s.append(f"{k} {a.mean():.3f}/{np.median(a):.3f}")
    print(name.ljust(16), " | ".join(s))
# 配对：咬合面模型 − 线上
for name in ("咬合面模型", "咬合面模型+加深1"):
    for k in ("iou", "corr", "cusp", "fossa", "cd_l2"):
        dd = np.array([r[f"{name}|{k}"] - r[f"线上(加深2)|{k}"] for r in rows if f"{name}|{k}" in r and f"线上(加深2)|{k}" in r], float)
        dd = dd[np.isfinite(dd)]
        print(f"{name} − 线上 {k}: 平均 {dd.mean():+.3f} ± {1.96 * dd.std(ddof=1) / np.sqrt(len(dd)):.3f}  变好 {np.mean(dd > 0 if k in ('iou', 'corr') else dd < 0):.0%}")
print("OCCL_EVAL_DONE")
