"""颈缘线训练数据：口扫上的模拟预备体。代型上训的模型搬到口扫上几乎用不了（预备体下面是牙龈，不是代型的底座），
所以在真实口扫（Teeth3DS 训练集）上按临床范围随机磨牙（synthetic.simulate_preparation），裁出预备体 + 周围 3 mm 牙龈，
真值颈缘线已知。CPU 多进程读口扫、磨牙、裁剪、算法向和凸棱；每个点到真值线的距离（训练目标）在 GPU 上成批算。

    python -m dental_ai.margin.scan_data --zips ../data/drive/raw/Teeth3DS --split <training_lower.txt> <training_upper.txt> --out ../data/margin_scan --per-jaw 3

输出 <out>/<扫描>_<牙号>.npz：verts、faces、feats (V, 7)、target (V,)、line（真值颈缘线）。margin/learned.py 的 train 直接能读。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CLIP = 2.0  # 和 learned.py 一致


def _one(args):
    """CPU 进程：一个颌 → 若干颗模拟预备体的裁剪块（还没有训练目标，只有“在预备体一侧”的符号）。"""
    item, per_jaw, seed, reach = args
    import trimesh
    from scipy.spatial import cKDTree

    from ..margin.detect import edge_sharpness
    from ..synthetic import simulate_preparation

    out = []
    try:
        mesh, labels = item.load()
        labels = np.asarray(labels)
        jaw = "upper" if item.name.endswith("upper") else "lower"
    except Exception as exc:  # noqa: BLE001
        return [{"name": item.name, "error": f"{type(exc).__name__}: {exc}"[:150]}]
    rng = np.random.default_rng(seed)
    present = sorted({int(f) for f in np.unique(labels) if f and 3 <= f % 10 <= 7 and (labels == f).sum() > 400})
    for fdi in rng.permutation(present)[:per_jaw].tolist():
        try:
            prm = dict(margin_above_gum=rng.uniform(0.3, 1.2), axial=rng.uniform(0.8, 1.4), occlusal=rng.uniform(1.2, 2.0),
                       taper_deg=rng.uniform(4, 10), scallop=rng.uniform(0.3, 0.6))
            prep, truth = simulate_preparation(mesh, labels, jaw, fdi, **prm)
            v0, v = np.asarray(mesh.vertices), np.asarray(prep.vertices)
            moved = np.linalg.norm(v - v0, axis=1) > 1e-6
            tooth = labels == fdi
            near = cKDTree(v[tooth]).query(v, distance_upper_bound=reach)[0] <= reach
            keep_v = tooth | (near & (labels == 0))
            keep_f = keep_v[np.asarray(prep.faces)].all(axis=1)
            crop = prep.submesh([np.flatnonzero(keep_f)], append=True, repair=False)
            parts = crop.split(only_watertight=False)
            if len(parts) > 1:
                crop = max(parts, key=lambda m: len(m.faces))
            cv = np.asarray(crop.vertices)
            # 裁剪块的顶点对回原网格：是不是磨过的那一侧（预备体）
            idx = cKDTree(v).query(cv)[1]
            side = np.where(moved[idx], 1.0, -1.0).astype(np.float32)
            feats = np.concatenate([cv, crop.vertex_normals, edge_sharpness(crop)[:, None]], axis=1).astype(np.float32)
            out.append({"name": f"{item.name}_{fdi}", "verts": cv.astype(np.float32), "faces": np.asarray(crop.faces, np.int32),
                        "feats": feats, "side": side, "line": truth.astype(np.float32), "params": prm})
        except Exception as exc:  # noqa: BLE001
            out.append({"name": f"{item.name}_{fdi}", "error": f"{type(exc).__name__}: {exc}"[:150]})
    return out


def targets_gpu(batch, device="cuda"):
    """GPU：每个点到真值颈缘线的最近距离（成批），target = 符号 × min(d, CLIP) / CLIP。"""
    import torch

    res = []
    for s in batch:
        p = torch.as_tensor(s["verts"], device=device)
        line = torch.as_tensor(s["line"], device=device)
        d = torch.cat([torch.cdist(p[i:i + 20000], line).min(dim=1).values for i in range(0, len(p), 20000)])
        t = torch.as_tensor(s["side"], device=device) * d.clamp(max=CLIP) / CLIP
        res.append(t.cpu().numpy().astype(np.float32))
    return res


def main() -> None:
    import multiprocessing as mp

    import torch

    from ..datasets import teeth3ds_zip_scans

    p = argparse.ArgumentParser()
    p.add_argument("--zips", type=Path, required=True)
    p.add_argument("--split", type=Path, nargs="*")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-jaw", type=int, default=3)
    p.add_argument("--reach", type=float, default=3.0)
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    items = [s for sp in (args.split or [None]) for s in teeth3ds_zip_scans(args.zips, sp)][: args.limit]
    jobs = [(it, args.per_jaw, i, args.reach) for i, it in enumerate(items)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0, n_ok, errors, pending = time.time(), 0, [], []

    def flush():
        nonlocal n_ok
        for s, t in zip(pending, targets_gpu(pending, device)):
            np.savez_compressed(args.out / f"{s['name']}.npz", verts=s["verts"], faces=s["faces"], feats=s["feats"], target=t, line=s["line"],
                                params=json.dumps(s["params"]))
            n_ok += 1
        pending.clear()

    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_one, jobs)):
            for s in res:
                if "error" in s:
                    errors.append(s)
                else:
                    pending.append(s)
            if len(pending) >= 64:
                flush()
            if (i + 1) % 100 == 0:
                print(json.dumps({"jaws": i + 1, "samples": n_ok + len(pending), "errors": len(errors), "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
    flush()
    print(json.dumps({"jaws": len(jobs), "samples": n_ok, "errors": len(errors), "error_examples": [e["error"] for e in errors[:5]],
                      "minutes": round((time.time() - t0) / 60, 1)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
