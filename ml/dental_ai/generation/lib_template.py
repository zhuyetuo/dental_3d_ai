"""形状模型的牙位模板：用对齐好的库牙（data/radial_lib，设计级牙形）按牙位求平均径向深度图，铺到穹顶模板的拓扑上。
每个方位角上，穹顶从顶点到边界圈的极角按“这个方位上牙冠最远到哪”等比例拉伸，所以边界圈正好落在牙冠边缘（牙颈线）上。
坐标和形状模型一致：原点 = 两侧邻牙质心的中点（径向图的投影中心在它下面 CENTER_DROP），目标牙在左侧时 x 取反。

    python -m dental_ai.generation.lib_template build --out ../data/radial_lib/dome_templates.npy
    python -m dental_ai.generation.lib_template check --templates ../data/radial_lib/dome_templates.npy --cache ../data/gen_cache/bite_val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .radial import CENTER_DROP, MAX_POLAR, SIZE


def mean_maps(dirs, max_score: float = 0.5):
    """每个牙位类别（0–15）：库牙深度图的平均（只在至少一半样本有牙冠的像素上），返回 (16, S, S)，没有的像素 0。"""
    from .learned import target_class
    from .radial_library import lib_dir_for

    s = np.zeros((16, SIZE, SIZE))
    n = np.zeros((16, SIZE, SIZE))
    cnt = np.zeros(16)
    for d in dirs:
        for g in sorted(lib_dir_for(Path(d)).glob("*.npz")):
            z = np.load(g)
            keep = z["ok"] & (z["score"] <= max_score)
            for k in np.flatnonzero(keep):
                c = target_class(int(z["fdi"][k]))
                m = z["lib"][k].astype(np.float32)
                s[c] += m
                n[c] += m > 0
                cnt[c] += 1
    out = np.where(n >= 0.5 * cnt[:, None, None], s / np.maximum(n, 1), 0.0)
    return out.astype(np.float32), cnt


def dome_from_map(depth, rings: int = 40, segments: int = 64):
    """平均深度图 → 穹顶拓扑的顶点 (1 + rings×segments, 3) mm（形状模型坐标）。"""
    from scipy import ndimage

    valid = depth > 0.3
    lab, k = ndimage.label(valid)
    if k > 1:
        valid = lab == (np.argmax(ndimage.sum(valid, lab, range(1, k + 1))) + 1)
    valid = ndimage.binary_fill_holes(valid)

    def at(theta, phi):
        rho = theta / MAX_POLAR
        i = np.clip(np.floor((rho * np.cos(phi) * 0.5 + 0.5) * SIZE).astype(int), 0, SIZE - 1)
        j = np.clip(np.floor((rho * np.sin(phi) * 0.5 + 0.5) * SIZE).astype(int), 0, SIZE - 1)
        return i, j

    # 填满牙冠范围内的空洞（最近有效像素）
    idx = ndimage.distance_transform_edt(~(depth > 0.3) | ~valid, return_distances=False, return_indices=True)
    filled = np.where(valid, depth[idx[0], idx[1]], 0.0)
    filled = np.where(valid & (depth > 0.3), depth, filled)
    origin = np.array([0.0, 0.0, -CENTER_DROP])
    verts = []
    thetas = np.linspace(0, MAX_POLAR, 400)
    for ring in range(rings + 1):
        for seg in range(segments if ring else 1):
            phi = 2 * np.pi * seg / segments
            if ring == 0:
                th = 0.0
            else:
                ii, jj = at(thetas, np.full_like(thetas, phi))
                inside = valid[ii, jj]
                th_max = thetas[np.flatnonzero(inside)[-1]] if inside.any() else np.deg2rad(60)
                th = th_max * ring / rings * 0.995
            i, j = at(np.array([th]), np.array([phi]))
            r = float(filled[i[0], j[0]])
            dvec = np.array([np.sin(th) * np.cos(phi), np.sin(th) * np.sin(phi), np.cos(th)])
            verts.append(origin + dvec * r)
    return np.array(verts, np.float32)


def build(dirs, out: Path):
    maps, cnt = mean_maps(dirs)
    temps = np.zeros((16, 1 + 40 * 64, 3), np.float32)
    for c in range(16):
        if cnt[c] < 20 or (maps[c] > 0).sum() < 500:
            continue
        temps[c] = dome_from_map(maps[c])
    np.save(out, temps)
    print(json.dumps({"out": str(out), "per_class_samples": cnt.astype(int).tolist(),
                      "extent_mm": [np.round(t.max(0) - t.min(0), 1).tolist() if np.abs(t).sum() else None for t in temps]}, ensure_ascii=False))


def check(templates: Path, cache: Path, n: int = 400):
    """坐标系核对：每个类别的模板到形状模型训练数据里真牙的倒角距离；同时试 z 取反、x 取反，看哪个最小。"""
    import torch

    from .learned import _dataset, chamfer, load_cache

    temps = torch.tensor(np.load(templates))
    scans, _ = load_cache(cache)
    items = [(s, t) for s, sc in enumerate(scans) for t in range(len(sc[1])) if sc[1][t] % 10 <= 7][:n]
    ds = _dataset(scans, items, False, 0.0, False)
    res = {"as_is": [], "flip_x": [], "flip_z": []}
    for k in range(len(items)):
        ctx, ctx_idx, cls, gt, *_ = ds[k]
        gt = torch.as_tensor(gt)[None].float()
        t = temps[int(cls)][None]
        if t.abs().sum() == 0:
            continue
        res["as_is"].append(float(chamfer(t, gt)))
        res["flip_x"].append(float(chamfer(t * torch.tensor([-1.0, 1, 1]), gt)))
        res["flip_z"].append(float(chamfer(t * torch.tensor([1.0, 1, -1]), gt)))
    print(json.dumps({k: round(float(np.median(v)), 3) for k, v in res.items()} | {"n": len(res["as_is"])}))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", type=Path, nargs="+", default=[Path("../data/radial/train"), Path("../data/radial/bite")])
    b.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("check")
    c.add_argument("--templates", type=Path, required=True)
    c.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "build":
        build(args.data, args.out)
    else:
        check(args.templates, args.cache)


if __name__ == "__main__":
    main()
