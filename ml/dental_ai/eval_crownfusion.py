"""真实预备体评分卡（FDI 16 Crown Dataset，CrownFusion 论文的数据）：真实磨好的 16 号预备体 + 技师设计的牙冠 + 邻牙 15/17 + 对颌 45–47。

形状模型（generation/learned.py）在这套数据的坐标里直接出牙冠外表面，和技师牙冠比：
- 外形：双向平均距离（mm）、双向 95% 距离；
- 颈缘：生成牙冠底边离预备体颈缘线（预备体网格的开边界）多远。
数据坐标：y 朝咬合面（对颌在 +y），x 从近中（15）指向远中（17），单位 mm。
我们的标准坐标（canonical_rotation）：牙冠 +z、往磨牙方向 +y、上颌患者右侧 +x。所以 标准 = (s·z, x, y)，s = ±1 由 --side 选（s=+1 是正常旋转）。

    python -m dental_ai.eval_crownfusion --data ~/zyt/dental_data/crownfusion/CrownFusion --split test \
        --model ../models/generation/toothgen_margin.pt --out ../runs/crownfusion/shape_margin
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def load_case(d: Path):
    pts = np.fromfile(d / "input_points.dat", np.float32).reshape(-1, 6)
    props = np.fromfile(d / "input_properties.dat", np.int8).reshape(-1, 3)
    prep = trimesh.load(d / "prep_mesh.ply", process=False)
    crown = trimesh.load(d / "outer_crown.ply", process=False)
    return pts, props, prep, crown


def margin_loop(prep: trimesh.Trimesh) -> np.ndarray:
    """预备体网格最长的开边界环（按顺序的点）= 颈缘线。"""
    from .margin.cervical import boundary_loops

    loops = boundary_loops(prep)
    loop = max(loops, key=len)
    return np.asarray(prep.vertices)[loop]


def surface_points(mesh: trimesh.Trimesh, n: int, seed: int = 0) -> np.ndarray:
    p, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(p)


def rim_points(mesh: trimesh.Trimesh) -> np.ndarray:
    from trimesh.grouping import group_rows

    e = mesh.edges_sorted
    return np.asarray(mesh.vertices)[np.unique(e[group_rows(e, require_count=1)])]


def occlusal_metrics(mesh: trimesh.Trimesh, ant: np.ndarray, seed: int = 0) -> dict:
    """mesh：牙冠（和 ant 同一坐标）；ant (K,6)：对颌点 + 朝向牙冠一侧的法向（antagonist_points 的约定，沿法向为正 = 在对颌外面）。"""
    rng = np.random.default_rng(seed)
    a = ant[rng.choice(len(ant), 30000, replace=len(ant) < 30000)]
    surf = surface_points(mesh, 60000, seed)
    d = np.sort(cKDTree(surf).query(a[:, :3])[0])
    bite10 = float(d[: max(1, len(d) // 10)].mean())
    tree = cKDTree(ant[:, :3])

    def signed(pts):
        dist, i = tree.query(pts)
        sd = ((pts - ant[i, :3]) * ant[i, 3:]).sum(1)
        return np.where(dist < 3.0, sd, np.inf)  # 离对颌 3 mm 以外的不算

    sd1k = signed(surface_points(mesh, 1000, seed + 1))
    inside = signed(np.asarray(mesh.triangles_center)) < 0
    return {"bite10": bite10, "inter_n": int((sd1k < 0).sum()), "inter_area": float(mesh.area_faces[inside].sum())}


def prep_points(prep: trimesh.Trimesh, n: int, seed: int) -> np.ndarray:
    """预备体表面取 n 个点 + 朝外的法向 (n,6)，数据坐标。"""
    p, fi = trimesh.sample.sample_surface(prep, n, seed=seed)
    nrm = np.asarray(prep.face_normals)[fi]
    if ((p - p.mean(0)) * nrm).sum(1).mean() < 0:  # 法向朝里就翻过来
        nrm = -nrm
    return np.concatenate([p, nrm], 1)


def run_case(model, d: Path, s: float, use_margin: bool, use_ant: bool, device: str, use_prep: bool = True):
    import torch

    from .generation.learned import (RIM_POINTS, TOOTH_POINTS, antagonist_points, build_context, crop_antagonist,
                                     target_class)
    from .margin.detect import resample_loop

    pts, props, prep, crown = load_case(d)
    R = np.array([[0, 0, s], [1, 0, 0], [0, 1, 0]], float)  # 行 = 标准坐标轴在数据坐标里的方向
    xyz, nrm = pts[:, :3].astype(float) @ R.T, pts[:, 3:].astype(float) @ R.T
    rng = np.random.default_rng(0)
    fdi, tp = [], []
    for side, f in ((1, 15), (-1, 17)):
        sel = (props[:, 1] == 0) & (props[:, 0] == side)
        if sel.sum() >= 200:
            fdi.append(f)
            tp.append(xyz[sel][rng.choice(int(sel.sum()), TOOTH_POINTS, replace=sel.sum() < TOOTH_POINTS)])
    if not fdi:
        return None
    fdi, tp = np.array(fdi), np.stack(tp).astype(np.float32)
    c, c_idx, origin = build_context(fdi, tp, 16, rng)
    extra = ()
    if getattr(model, "antagonist", False):
        a6 = None
        ant = props[:, 1] == 1
        if use_ant and ant.sum() >= 64:
            a6 = crop_antagonist(antagonist_points(xyz[ant], nrm[ant]), origin, rng)
        extra = (None, None) if a6 is None else (torch.tensor(a6.astype(np.float32))[None].to(device), torch.ones(1, device=device))
    margin = margin_loop(prep)
    if getattr(model, "margin", False):
        if not extra:
            extra = (None, None)
        if use_margin:
            mp = resample_loop(margin, 4 * RIM_POINTS) @ R.T - origin
            mp = mp[rng.choice(len(mp), RIM_POINTS, replace=False)]
            extra = extra + (torch.tensor(mp.astype(np.float32))[None].to(device), torch.ones(1, device=device))
    if getattr(model, "prep", False) and use_prep:
        extra = extra + (None,) * (4 - len(extra))  # 位置参数：对颌、有没有、颈缘、有没有，然后才是预备体
        p6 = prep_points(prep, 1024, 0)
        p6 = np.concatenate([p6[:, :3] @ R.T - origin, p6[:, 3:] @ R.T], 1)
        extra = extra + (torch.tensor(p6.astype(np.float32))[None].to(device), torch.ones(1, device=device))
    with torch.no_grad():
        pred = model(torch.tensor(c.astype(np.float32))[None].to(device), torch.tensor(c_idx.astype(np.int64))[None].to(device),
                     torch.tensor([target_class(16)]).to(device), *extra)[0].cpu().numpy().astype(float)
    verts = (pred + origin) @ R  # 回到数据坐标
    gen = trimesh.Trimesh(verts, model.faces.cpu().numpy(), process=False)
    a, b = surface_points(gen, 20000), surface_points(crown, 20000)
    d_ab, d_ba = cKDTree(b).query(a)[0], cKDTree(a).query(b)[0]
    ring = verts[model.boundary.cpu().numpy()]
    # CrownFusion 代码里的 CD-L2：两个方向“最近点距离平方”的均值相加再开方（各 1 万点）
    a10, b10 = surface_points(gen, 10000, 1), surface_points(crown, 10000, 1)
    cf = float(np.sqrt((cKDTree(b10).query(a10)[0] ** 2).mean() + (cKDTree(a10).query(b10)[0] ** 2).mean()))
    occ = {}
    antm = props[:, 1] == 1
    if antm.sum() >= 64:  # 咬合指标在标准坐标里算（对颌法向约定和形状模型训练时一样）
        a6 = antagonist_points(xyz[antm], nrm[antm])
        g = occlusal_metrics(trimesh.Trimesh(verts @ R.T, gen.faces, process=False), a6)
        t = occlusal_metrics(trimesh.Trimesh(np.asarray(crown.vertices) @ R.T, crown.faces, process=False), a6)
        occ = {**g, **{f"gt_{k}": v for k, v in t.items()}}
    return {"case": d.name, "neighbors": len(fdi), "antagonist": bool(extra and extra[0] is not None), "cd_l2_cf": cf, **occ,
            "cd": float((d_ab.mean() + d_ba.mean()) / 2), "gen_to_gt": float(d_ab.mean()), "gt_to_gen": float(d_ba.mean()),
            "p95": float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95))),
            "rim_to_margin": float(cKDTree(margin).query(ring)[0].mean()),
            "gt_rim_to_margin": float(cKDTree(margin).query(rim_points(crown))[0].mean()),
            "height_err": float((verts @ R.T)[:, 2].max() - (np.asarray(crown.vertices) @ R.T)[:, 2].max())}, gen


def _cache_case(args):
    """一例 → 形状模型训练缓存（generation/learned.py 的格式，标准坐标）：邻牙 15/17 取输入点，16 号取技师牙冠外表面的点，
    16 号的“边界环”换成预备体颈缘线（训练时模型的底边对齐它、颈缘线输入也是它加扰动，和推理时给的一致），对颌带法向。
    target=16：只把 16 号当目标（邻牙没有完整的边界，不当目标）。"""
    from .generation.learned import RIM_POINTS, TOOTH_POINTS, antagonist_points
    from .margin.detect import resample_loop

    d, out, s = args
    dest = out / f"cf{d.name}.npz"
    if dest.exists():
        return True
    try:
        pts, props, prep, crown = load_case(d)
        R = np.array([[0, 0, s], [1, 0, 0], [0, 1, 0]], float)
        xyz, nrm = pts[:, :3].astype(float) @ R.T, pts[:, 3:].astype(float) @ R.T
        rng = np.random.default_rng(int(d.name))
        fdi, tp, rim = [], [], []
        for side, f in ((1, 15), (-1, 17)):
            sel = np.flatnonzero((props[:, 1] == 0) & (props[:, 0] == side))
            if len(sel) >= 200:
                fdi.append(f)
                tp.append(xyz[rng.choice(sel, TOOTH_POINTS, replace=len(sel) < TOOTH_POINTS)])
                rim.append(np.zeros((RIM_POINTS, 3)))
        if not fdi:
            return False
        fdi.append(16)
        tp.append(surface_points(crown, TOOTH_POINTS, seed=int(d.name)) @ R.T)
        m = resample_loop(margin_loop(prep), 4 * RIM_POINTS) @ R.T
        rim.append(m[rng.choice(len(m), RIM_POINTS, replace=False)])
        o = np.argsort(fdi)
        ant = props[:, 1] == 1
        a6 = antagonist_points(xyz[ant], nrm[ant]) if ant.sum() >= 64 else np.zeros((0, 6), np.float32)
        p6 = prep_points(prep, 4096, int(d.name))
        p6 = np.concatenate([p6[:, :3] @ R.T, p6[:, 3:] @ R.T], 1)  # 标准坐标
        np.savez_compressed(dest, fdi=np.array(fdi, np.int16)[o], pts=np.stack(tp).astype(np.float32)[o],
                            rim=np.stack(rim).astype(np.float32)[o], ant=a6.astype(np.float16), target=np.int16(16),
                            prep=p6.astype(np.float16))
        return True
    except Exception as exc:  # noqa: BLE001
        print("skip", d.name, type(exc).__name__, exc, flush=True)
        return False


def build_cache(data: Path, split: str, out: Path, s: float, workers: int) -> int:
    import multiprocessing as mp

    ids = json.load(open(data / {"test": "testing.json", "val": "validation.json", "train": "training.json"}[split]))
    out.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        ok = sum(pool.imap_unordered(_cache_case, [(data / split / str(c), out, s) for c in ids], chunksize=8))
    return ok


def summarize(rows):
    keys = ["cd_l2_cf", "cd", "gen_to_gt", "gt_to_gen", "p95", "rim_to_margin", "gt_rim_to_margin", "height_err",
            "bite10", "gt_bite10", "inter_n", "gt_inter_n", "inter_area", "gt_inter_area"]
    out = {"n": len(rows)}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r])
        if not len(v):
            continue
        out[k] = {"median": round(float(np.median(v)), 3), "mean": round(float(v.mean()), 3)}
    out["with_antagonist"] = sum(r["antagonist"] for r in rows)
    out["neighbors"] = {n: sum(r["neighbors"] == n for r in rows) for n in (1, 2)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--side", type=float, default=1.0)
    ap.add_argument("--no-margin", action="store_true")
    ap.add_argument("--no-antagonist", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--save", type=int, default=12, help="存前几例生成的牙冠（画图用）")
    ap.add_argument("--build-cache", action="store_true", help="不评测，把这个划分做成形状模型的训练缓存，写到 --out")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    if a.build_cache:
        print("cached", build_cache(a.data, a.split, a.out, a.side, a.workers))
        return
    import torch

    from .generation.learned import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(Path(a.model), device)
    model.eval()
    ids = json.load(open(a.data / {"test": "testing.json", "val": "validation.json", "train": "training.json"}[a.split]))
    ids = sorted(ids, key=int)[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, cid in enumerate(ids):
        try:
            r = run_case(model, a.data / a.split / str(cid), a.side, not a.no_margin, not a.no_antagonist, device)
        except Exception as exc:  # noqa: BLE001
            print("skip", cid, type(exc).__name__, exc, flush=True)
            continue
        if r is None:
            continue
        row, gen = r
        rows.append(row)
        if i < a.save:
            gen.export(a.out / f"{cid}_gen.ply")
    (a.out / "rows.json").write_text(json.dumps(rows))
    s = summarize(rows)
    (a.out / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1))
    print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
