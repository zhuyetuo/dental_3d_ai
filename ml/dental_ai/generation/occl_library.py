"""技师咬合面库：从 9513 颗训练集技师牙冠里检索最像的几颗，非刚性对齐后借它的咬合面起伏（牙尖、嵴、窝、沟成套），换掉我们自己那一段。

上游（体素模型 0.156 mm 一格）出来的咬合面是“平均的”：牙尖圆平、窝浅、嵴不明显。单独刻沟线反而像划痕（用户看图判断）。
这里不改整体形状（> 1 mm 的起伏不动），只把比 1 mm 细的起伏换成真实技师设计的。
40 例真实预备体（16 号）：整体形状 CD-L2 不变（+0.003），细沟幅度 0.037 → 0.05（技师 0.056），看起来像真牙；
窝沟位置和本例技师牙冠的吻合度略低（借来的是另一位技师的细节）。

库文件（models/generation/occl_library/，build 子命令从 data/occl_cache/train 生成）：
- keys.npz：检索键（主窝沟图 + 咬合面形状，64×64）、有效区；
- heights.npy：技师咬合面高度图（256×256，float16，memmap，只读用到的几颗）；masks.npy：有效区（packbits）。
坐标：数据坐标（y 朝咬合面），高度图和 occlusal 一样（0.055 mm 一格）。

    python -m dental_ai.generation.occl_library build --cache ../data/occl_cache/train --out ../models/generation/occl_library
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from . import occlusal as oc

DS = 4  # 256 → 64 的检索小图
RELIEF_MM = 1.0  # 借比这个细的起伏
_LIB = {}


def _detrend(a, m):
    yy, xx = np.nonzero(m)
    if len(yy) < 10:
        return a
    A = np.c_[xx, yy, np.ones(len(xx))]
    c = np.linalg.lstsq(A, a[m], rcond=None)[0]
    return a - (c[0] * np.arange(a.shape[1])[None, :] + c[1] * np.arange(a.shape[0])[:, None] + c[2])


def key_of(h, m):
    """检索键：主窝沟图 + 去掉倾斜平面的低频咬合面高度，下采样到 64×64、有效区内标准化。"""
    g = oc.groove_depth_map(h, m)
    low = _detrend(oc.lowpass(h, m, 0.6), m)
    mm = m.reshape(64, DS, 64, DS).mean((1, 3)) > 0.5
    out = []
    for a in (g, low):
        a = np.where(m, a, 0.0).reshape(64, DS, 64, DS).mean((1, 3))
        a = np.where(mm, a, 0.0)
        a = (a - a[mm].mean()) / (a[mm].std() + 1e-6) * mm
        out.append(a.astype(np.float32))
    return np.stack(out), mm


def build(cache: Path, out: Path) -> None:
    files = sorted(cache.glob("*.npz"))
    out.mkdir(parents=True, exist_ok=True)
    n = len(files)
    H = np.lib.format.open_memmap(out / "heights.npy", mode="w+", dtype=np.float16, shape=(n, oc.RES, oc.RES))
    Mb = np.zeros((n, oc.RES * oc.RES // 8), np.uint8)
    K = np.zeros((n, 2, 64, 64), np.float16)
    KM = np.zeros((n, 64, 64), bool)
    for i, f in enumerate(files):
        z = np.load(f)
        x, y, m = z["x"].astype(np.float32), z["y"].astype(np.float32), z["m"]
        mc = x[1] > 0
        ht = np.where(m, x[0] + y, x[0])
        mm = m & mc
        H[i] = np.where(mm, ht, 0).astype(np.float16)
        Mb[i] = np.packbits(mm.ravel())
        k, km = key_of(ht, mm)
        K[i], KM[i] = k.astype(np.float16), km
        if i % 1000 == 0:
            print("library", i, flush=True)
    H.flush()
    np.save(out / "masks.npy", Mb)
    np.savez_compressed(out / "keys.npz", K=K, M=KM, files=np.array([f.name for f in files]))
    print("library done", n, flush=True)


def _load(path: Path):
    key = str(path)
    if key not in _LIB:
        import torch

        kz = np.load(path / "keys.npz")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _LIB[key] = {"K": torch.tensor(kz["K"].astype(np.float32), device=dev), "M": torch.tensor(kz["M"], device=dev).float(),
                     "H": np.load(path / "heights.npy", mmap_mode="r"), "Mb": np.load(path / "masks.npy", mmap_mode="r"), "dev": dev}
    return _LIB[key]


def retrieve_topk(lib, h, m, k=5, shifts=range(-3, 4)):
    """最像的 k 颗：两张小图的重叠区相关，允许 ±3 小格（±0.66 mm）平移。返回 [(索引, (dy, dx) 格)]。"""
    import torch

    q, qm = key_of(h, m)
    q = torch.tensor(q, device=lib["dev"])
    qm = torch.tensor(qm, device=lib["dev"]).float()
    K, Mk = lib["K"], lib["M"]
    best = torch.full((len(K),), -1e9, device=lib["dev"])
    bsh = torch.zeros((len(K), 2), dtype=torch.long, device=lib["dev"])
    for dy in shifts:
        for dx in shifts:
            qs = torch.roll(q, (dy, dx), (1, 2))
            qms = torch.roll(qm, (dy, dx), (0, 1))
            ov = Mk * qms
            s = ((K * qs[None]).sum(1) * ov).sum((1, 2)) / (2 * ov.sum((1, 2)).clamp(min=1))
            s = s * (ov.sum((1, 2)) / qms.sum().clamp(min=1)).clamp(max=1)
            better = s > best
            best = torch.where(better, s, best)
            bsh[better] = torch.tensor([dy * DS, dx * DS], device=lib["dev"])
    _, idx = best.topk(k)
    return [(int(i), tuple(int(t) for t in bsh[i].tolist())) for i in idx.tolist()]


def _mid(h, m):
    return np.where(m, oc.lowpass(h, m, 0.3) - oc.lowpass(h, m, 2.0), 0.0)


def demons(f, mov, mask, iters=120, smooth=4.0, maxdisp=20):
    """2D 非刚性配准（demons）：mov∘(grid+u) ≈ f，u 平滑、最多 maxdisp 格（1.1 mm）。"""
    from scipy import ndimage

    H, W = f.shape
    gy, gx = np.mgrid[0:H, 0:W].astype(float)
    u = np.zeros((2, H, W))
    fs, ms = ndimage.gaussian_filter(f, 1.5), ndimage.gaussian_filter(mov, 1.5)
    for _ in range(iters):
        mw = ndimage.map_coordinates(ms, [gy + u[0], gx + u[1]], order=1, mode="constant")
        diff = (mw - fs) * mask
        dy, dx = np.gradient(mw)
        den = dx ** 2 + dy ** 2 + diff ** 2 + 1e-9
        u[0] -= diff * dy / den
        u[1] -= diff * dx / den
        u = np.stack([ndimage.gaussian_filter(u[0], smooth), ndimage.gaussian_filter(u[1], smooth)])
        mag = np.sqrt((u ** 2).sum(0))
        u *= np.minimum(1, maxdisp / np.maximum(mag, 1e-9))[None]
    return u


def _warp(a, u):
    from scipy import ndimage

    H, W = a.shape
    gy, gx = np.mgrid[0:H, 0:W].astype(float)
    return ndimage.map_coordinates(a, [gy + u[0], gx + u[1]], order=1, mode="constant")


def _ncc(a, b, m):
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def relief(h, m, R=RELIEF_MM):
    return oc.lowpass(h, m, 0.055) - oc.lowpass(h, m, R)


def borrow(verts, faces, lib_path: Path, k: int = 5, gain: float | None = None, fade=None):
    """牙冠外表面（数据坐标）→ 借了技师咬合面起伏的顶点（面不变）、信息。库不在时原样返回。"""
    import trimesh
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from trimesh.grouping import group_rows

    lib_path = Path(lib_path)
    if not (lib_path / "keys.npz").exists():
        return verts, {"occl_library": "missing"}
    gain = float(os.environ.get("OCCL_LIB_GAIN", 1.2)) if gain is None else gain
    f0, f1 = fade if fade is not None else (1.2, 1.0)
    lib = _load(lib_path)
    m0 = trimesh.Trimesh(np.asarray(verts, float), np.asarray(faces), process=False)
    pts = np.asarray(trimesh.sample.sample_surface(m0, 300000, seed=1)[0])
    x4, (cx, cz, top) = oc.inputs(pts, None)
    mc = x4[1] > 0
    h0 = x4[0]
    f_mid = _mid(h0, mc)
    best = None
    for i, (sy, sx) in retrieve_topk(lib, h0, mc, k):
        zm = np.unpackbits(np.asarray(lib["Mb"][i]))[: oc.RES * oc.RES].reshape(oc.RES, oc.RES).astype(bool)
        zh = np.asarray(lib["H"][i], np.float32)
        zm2, zh2 = np.roll(zm, (-sy, -sx), (0, 1)), np.roll(zh, (-sy, -sx), (0, 1))
        both = mc & zm2
        if both.sum() < 1000:
            continue
        mm_ = _mid(zh2, zm2)
        u = demons(f_mid, mm_, both.astype(float))
        s = _ncc(f_mid, _warp(mm_, u), both)
        if best is None or s > best[0]:
            best = (s, i, zh2, zm2, u)
    if best is None:
        return verts, {"occl_library": "no_match"}
    s, i, zh2, zm2, u = best
    zmw = _warp(zm2.astype(float), u) > 0.5
    tw = _warp(relief(zh2, zm2), u)
    inner = ndimage.distance_transform_edt(mc & zmw) * oc.CELL
    fd = np.clip((inner - f0) / f1, 0, 1)
    dmap = (gain * tw - relief(h0, mc)) * fd
    v = np.asarray(verts, float).copy()
    n = m0.vertex_normals
    fx = (v[:, 0] - cx) / oc.CELL + oc.RES / 2 - 0.5
    fz = (v[:, 2] - cz) / oc.CELL + oc.RES / 2 - 0.5
    dv = ndimage.map_coordinates(dmap, [fx, fz], order=1, mode="constant", cval=0.0)
    w = np.clip((n[:, 1] - 0.3) / 0.3, 0, 1) * (v[:, 1] >= top - oc.TOP_BAND)
    es = m0.edges_sorted
    rim = np.unique(es[group_rows(es, require_count=1)])
    if len(rim):
        w *= np.clip(cKDTree(v[rim]).query(v)[0] / 1.0, 0, 1)
    v[:, 1] += dv * w
    if os.environ.get("OCCL_LIB_TIDY", "1") == "1":  # 陡坡上竖直挪起伏会挤出尖棱 / 翻折（16 号示例 2400+ 处）：只在借了起伏的地方局部 Taubin
        v = _crease_taubin(v, np.asarray(faces), (w > 0.05) & (np.abs(dv) > 1e-4))
    return v, {"occl_library": int(i), "occl_library_ncc": round(s, 3)}


def _crease_taubin(v, f, movable, rounds=6, ring=2, iters=4, thr=0.0):
    """尖棱（相邻面法向点积 < thr）周围几圈、movable 的顶点做 Taubin 平滑（不缩体积）。返回新顶点。"""
    import trimesh

    v = np.asarray(v, float).copy()
    m = trimesh.Trimesh(v, f, process=False)
    e = m.edges_unique
    n = len(v)
    deg = np.maximum(np.bincount(e.ravel(), minlength=n), 1).astype(float)[:, None]
    for _ in range(rounds):
        m = trimesh.Trimesh(v, f, process=False)
        fn = m.face_normals
        adj = m.face_adjacency
        bad = (fn[adj[:, 0]] * fn[adj[:, 1]]).sum(1) < thr
        if not bad.any():
            break
        act = np.zeros(n, bool)
        act[f[adj[bad].ravel()].ravel()] = True
        for _ in range(ring):
            a2 = act.copy()
            a2[e[act[e[:, 0]], 1]] = True
            a2[e[act[e[:, 1]], 0]] = True
            act = a2
        wt = (act & movable)[:, None].astype(float)
        for _ in range(iters):
            for s_ in (0.5, -0.53):
                acc = np.empty_like(v)
                for k in range(3):
                    acc[:, k] = np.bincount(e[:, 0], v[e[:, 1], k], n) + np.bincount(e[:, 1], v[e[:, 0], k], n)
                v = v + s_ * (acc / deg - v) * wt
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--cache", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    build(a.cache, a.out)


if __name__ == "__main__":
    main()
