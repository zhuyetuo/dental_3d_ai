"""小窝沟是没学到，还是被平滑掉了？每一步的咬合面高度图按尺度拆成“大窝沟”（0.6–2 mm）和“小窝沟”（0.12–0.6 mm）两段，
量各段的起伏幅度（RMS）和与技师同一段的相关。步骤：体素模型的点 → 泊松表面 → 线上最终网格 → 技师；另加生成式咬合面模型（高度图上）。
    python band_analysis.py 40"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from trimesh.grouping import group_rows

sys.path.insert(0, str(Path.home() / "zyt/dental_3d_ai/ml"))
from dental_ai.eval_crownfusion import load_case, run_case, surface_points  # noqa: E402
from dental_ai.generation import occlusal as oc  # noqa: E402
from dental_ai.generation import voxel_crown as vc  # noqa: E402
from dental_ai.generation.learned import load_model  # noqa: E402
from dental_ai.generation.poisson import deepen_grooves, poisson_mesh  # noqa: E402

DATA = Path.home() / "zyt/dental_data/crownfusion/CrownFusion"
ROOT = Path.home() / "zyt/dental_3d_ai"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
vox = vc.load(ROOT / "models/generation/voxel_crown.pt")
shape, _ = load_model(ROOT / "models/generation/toothgen_ctx.pt", "cuda")
flow = oc.load(ROOT / "runs/occl/occl_flow.pt")
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


SIG = (0.055, 0.25, 0.6, 2.0)  # 高斯 σ（mm）：细小窝沟 0.055–0.25、中等 0.25–0.6、大窝沟 0.6–2


def bands(h, m):
    ls = [oc.lowpass(h, m, s) for s in SIG]
    return ls[2] - ls[3], ls[1] - ls[2], ls[0] - ls[1]  # 大、中、细


res = {}
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
    x4, frame = oc.inputs(surface_points(pm, 300000, 1), ant)
    cx, cz, top = frame
    mc = x4[1] > 0
    ht, mt = oc.coarse_map(surface_points(crown, 300000, 2), cx, cz, top)
    # 体素点：朝上的点每格取中位高度
    up = p[(n[:, 1] > 0.3) & (p[:, 1] >= top - oc.TOP_BAND)]
    ix = np.floor((up[:, 0] - cx) / oc.CELL + oc.RES / 2).astype(int)
    iz = np.floor((up[:, 2] - cz) / oc.CELL + oc.RES / 2).astype(int)
    ok = (ix >= 0) & (ix < oc.RES) & (iz >= 0) & (iz < oc.RES)
    s_ = np.zeros((oc.RES, oc.RES)); c_ = np.zeros((oc.RES, oc.RES))
    np.add.at(s_, (ix[ok], iz[ok]), up[ok, 1]); np.add.at(c_, (ix[ok], iz[ok]), 1)
    hv, mv = oc._fill(np.where(c_ > 0, s_ / np.maximum(c_, 1), 0.0), c_ > 0, iters=3)
    det_c = x4[0] - oc.lowpass(x4[0], mc)
    x5 = np.concatenate([x4, det_c[None]]).astype(np.float32)
    with torch.no_grad():
        dd = oc.flow_sample(flow, oc._scale(torch.tensor(x5)[None].cuda()))[0].cpu().numpy()
    maps = {"体素模型的点": (hv, mv), "泊松表面": (x4[0] + top, mc),
            "线上最终网格": oc.coarse_map(surface_points(live, 300000, 3), cx, cz, top),
            "生成式咬合面(高度图)": (x4[0] + top - det_c + 2 * ndimage.gaussian_filter(dd, 1.5), mc), "技师": (ht, mt)}
    # 技师牙冠原网格“平滑掉”能损失多少：同样过一遍线上的 Taubin 3 次
    tm = crown.copy()
    trimesh.smoothing.filter_taubin(tm, lamb=0.5, nu=-0.53, iterations=3)
    maps["技师+线上平滑"] = oc.coarse_map(surface_points(tm, 300000, 2), cx, cz, top)
    allm = mt.copy()
    for h, m in maps.values():
        allm &= m
    v_ = ndimage.binary_erosion(allm, iterations=int(0.8 / oc.CELL))
    if v_.sum() < 200:
        continue
    np.savez_compressed(f"/tmp/band_maps_{cid}.npz", valid=v_, **{k: h for k, (h, m) in maps.items()})
    tb = bands(ht, mt)
    for k, (h, m) in maps.items():
        b = bands(h, m)
        res.setdefault(k, []).append([x[v_].std() for x in b] + [np.corrcoef(x[v_], t[v_])[0, 1] for x, t in zip(b, tb)])
    print(len(res["技师"]), flush=True)
print("n", len(res["技师"]))
print(f"{'':20s} 幅度 大 / 中 / 细(mm)       | 位置相关 大 / 中 / 细")
for k, s in res.items():
    a = np.nanmean(np.array(s), 0)
    print(f"{k:20s} {a[0]:.3f} / {a[1]:.3f} / {a[2]:.3f} | {a[3]:.2f} / {a[4]:.2f} / {a[5]:.2f}")
print("BAND_DONE")
