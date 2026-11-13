"""点到三角网格的最近点（精确），批量矩阵运算，有显卡时在显卡上算。

trimesh.proximity.closest_point 逐点查空间索引，口扫的对颌有几万个三角形、牙冠要查几万个点，一次要零点几秒到几秒，
咬合调整、邻接调整和评分卡（eval_crown）每颗牙冠要查几十次，60 位患者的评分卡要跑一个多小时，时间几乎都花在这里。

这里的做法：
1. 用三角形中心建 KD 树，每个点先取最近的 k 个三角形当候选；
2. 所有 (点, 候选三角形) 一起算精确的点到三角形最近点（torch 矩阵运算），取最小；
3. 核对结果是否一定正确：没进候选的三角形，中心离得比第 k 个候选还远，它到点的距离不会小于
   “第 k 个候选的中心距离 − 三角形最大半径”。已找到的最小距离不超过这个下界就是精确解；
   超过的点（多半离曲面较远）改查“中心在 已找到的距离 + 最大半径 以内”的全部三角形，最近的三角形一定在里面；
4. 少数特别大的三角形（半径超过 99.5% 分位）单独拿出来，对每个点都算，避免它们把“最大半径”撑大。

环境变量 DENTAL_MESHDIST=trimesh 退回 trimesh；DENTAL_MESHDIST_DEVICE=cpu 强制用 CPU。
"""

from __future__ import annotations

import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree

try:  # 没装 torch 时退回 trimesh
    import torch
except ImportError:  # pragma: no cover
    torch = None

_PAIRS = 500_000  # 一批最多算多少对 (点, 三角形)。一对要占几百字节的中间结果，多进程评估时每个进程都占一份显存
_state = {"device": None}
_WORKERS = int(os.environ.get("OMP_NUM_THREADS", "-1"))  # KD 树查询的线程数；多进程评估时由 OMP_NUM_THREADS 限制


def _device():
    if _state["device"] is None:
        want = os.environ.get("DENTAL_MESHDIST_DEVICE", "")
        name = want or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            torch.zeros(1, device=name)
        except Exception:  # 显卡不可用（驱动、fork 后的子进程等）
            name = "cpu"
        _state["device"] = torch.device(name)
    return _state["device"]


def _index(surface: trimesh.Trimesh) -> dict:
    """三角形中心的 KD 树、半径、大三角形。挂在 trimesh 自己的缓存上，顶点一变就失效。"""
    cache = surface._cache
    key = "dental_meshdist_index"
    if key in cache:
        return cache[key]
    tris = np.asarray(surface.triangles, dtype=np.float64)
    center = tris.mean(axis=1)
    radius = np.linalg.norm(tris - center[:, None], axis=2).max(axis=1)
    cut = float(np.quantile(radius, 0.995)) if len(radius) > 200 else float(radius.max())
    small = np.flatnonzero(radius <= cut)
    index = {
        "tris": tris,
        "normals": np.asarray(surface.face_normals, dtype=np.float64),
        "small": small,
        "big": np.flatnonzero(radius > cut),
        "tree": cKDTree(center[small]),
        "r_max": float(radius[small].max()),
    }
    cache[key] = index
    return index


def _closest_on_triangles(p, a, b, c):
    """点到三角形的最近点（Ericson, Real-Time Collision Detection 5.1.5），各参数形状 (..., 3)。"""
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    eps = 1e-30

    def safe(x):
        return torch.where(x.abs() < eps, torch.full_like(x, eps), x)

    denom = safe(va + vb + vc)
    out = a + ab * (vb / denom)[..., None] + ac * (vc / denom)[..., None]  # 落在三角形内部
    # 以下按 Ericson 的判断顺序，先判断的优先，所以倒着覆盖
    t = ((d4 - d3) / safe((d4 - d3) + (d5 - d6)))[..., None]
    out = torch.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None], b + (c - b) * t, out)  # 边 bc
    out = torch.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], a + ac * (d2 / safe(d2 - d6))[..., None], out)  # 边 ac
    out = torch.where(((d6 >= 0) & (d5 <= d6))[..., None], c, out)  # 顶点 c
    out = torch.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], a + ab * (d1 / safe(d1 - d3))[..., None], out)  # 边 ab
    out = torch.where(((d3 >= 0) & (d4 <= d3))[..., None], b, out)  # 顶点 b
    out = torch.where(((d1 <= 0) & (d2 <= 0))[..., None], a, out)  # 顶点 a
    return out


_TIE = 1e-9  # 距离相差这么多以内算“并列最近”（毫米）


def _best(points, tris, normals, cand):
    """points (P,3)、tris (F,3,3)、normals (F,3) 都是 torch 张量，cand (P,K) 是候选三角形编号（-1 = 补位）。
    返回每个点的 (最近点, 距离, 三角形编号, 并列最近的三角形的法向之和)。"""
    P, K = cand.shape
    step = max(1, _PAIRS // max(K, 1))
    cps, ds, ts, ns = [], [], [], []
    for s in range(0, P, step):
        c = cand[s : s + step]
        pad = c < 0
        c = c.clamp(min=0)
        t = tris[c]  # (p, K, 3, 3)
        p = points[s : s + step, None, :].expand(-1, K, -1)
        q = _closest_on_triangles(p, t[:, :, 0], t[:, :, 1], t[:, :, 2])
        d = torch.linalg.norm(q - p, dim=-1)
        d = torch.nan_to_num(d, nan=float("inf"))
        d[pad] = float("inf")
        low, j = d.min(dim=1)
        row = torch.arange(len(c), device=c.device)
        cps.append(q[row, j])
        ds.append(low)
        ts.append(c[row, j])
        tied = d <= low[:, None] + _TIE
        ns.append((normals[c] * tied[..., None]).sum(dim=1))
    return torch.cat(cps), torch.cat(ds), torch.cat(ts), torch.cat(ns)


def closest_point(surface: trimesh.Trimesh, points: np.ndarray, k: int = 16, return_normal: bool = False):
    """和 trimesh.proximity.closest_point 的返回值一样：(最近点 (P,3), 距离 (P,), 三角形编号 (P,))。

    return_normal=True 时多返回一项：最近点处的法向（单位向量）。最近点落在棱或顶点上时，几个三角形并列最近，
    只取其中一个的法向来判断点在曲面的哪一侧，结果取决于恰好选中了哪一个（凹棱、凸棱处两个法向给出的正负可以相反）；
    这里取并列最近的几个三角形法向之和，和选中哪一个无关。"""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if torch is None or os.environ.get("DENTAL_MESHDIST") == "trimesh" or len(points) == 0 or len(surface.faces) == 0:
        cp, dist, tri = trimesh.proximity.closest_point(surface, points)
        return (cp, dist, tri, np.asarray(surface.face_normals)[tri]) if return_normal else (cp, dist, tri)
    try:
        out = _closest_point(surface, points, k, _device())
    except RuntimeError as exc:  # 显存不够（多进程一起跑）：这一次在 CPU 上算
        if "out of memory" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        out = _closest_point(surface, points, k, torch.device("cpu"))
    return out if return_normal else out[:3]


def _closest_point(surface, points, k, dev):
    index = _index(surface)
    tris = torch.as_tensor(index["tris"], device=dev)
    normals = torch.as_tensor(index["normals"], device=dev)
    pts = torch.as_tensor(points, device=dev)
    small, big, tree, r_max = index["small"], index["big"], index["tree"], index["r_max"]
    n = len(points)
    cp = torch.zeros((n, 3), dtype=torch.float64, device=dev)
    dist = torch.full((n,), float("inf"), dtype=torch.float64, device=dev)
    tri = torch.zeros(n, dtype=torch.long, device=dev)
    normal = torch.zeros((n, 3), dtype=torch.float64, device=dev)

    def update(rows, cand, complete=False):
        """complete：候选里一定包含全部并列最近的三角形，距离一样时也换成这次的结果（法向之和才完整）。"""
        q, d, t, m = _best(pts[rows], tris, normals, cand)
        better = (d <= dist[rows] + _TIE) if complete else (d < dist[rows])
        r = rows[better]
        cp[r], dist[r], tri[r], normal[r] = q[better], d[better], t[better], m[better]

    everyone = torch.arange(n, device=dev)
    if len(big):  # 大三角形：每个点都算
        update(everyone, torch.as_tensor(big, device=dev)[None, :].expand(n, -1))
    if len(small):
        small_t = torch.as_tensor(small, device=dev)
        kk = min(k, len(small))
        dc, ic = tree.query(points, k=kk, workers=_WORKERS)
        if kk == 1:
            dc, ic = dc[:, None], ic[:, None]
        update(everyone, small_t[torch.as_tensor(ic, device=dev)])
        if kk < len(small):
            # 没进候选的三角形到点的距离不会小于 “第 k 个候选的中心距离 − 最大半径”；已找到的距离超过它的点还不能确定
            found = dist.cpu().numpy()
            todo = np.flatnonzero(found > dc[:, -1] - r_max)
            if len(todo):
                # 这些点（多半离曲面较远）改查“中心在 已找到的距离 + 最大半径 以内”的全部三角形：真正的最近三角形一定在里面
                lists = tree.query_ball_point(points[todo], found[todo] + r_max, workers=_WORKERS, return_sorted=False)
                count = np.array([len(x) for x in lists])
                order = np.argsort(count)
                s = 0
                while s < len(order):
                    width = int(count[order[s]])
                    # 候选数相近的点放一批，补齐到同样长度（补 -1）
                    e = s + 1
                    while e < len(order) and count[order[e]] <= max(width * 2, 32) and (e - s + 1) * count[order[e]] <= _PAIRS:
                        e += 1
                    width = int(count[order[e - 1]])
                    if width:
                        cand = np.full((e - s, width), -1, dtype=np.int64)
                        for i, j in enumerate(order[s:e]):
                            row = lists[j]
                            cand[i, : len(row)] = small[row]
                        update(torch.as_tensor(todo[order[s:e]], device=dev), torch.as_tensor(cand, device=dev), complete=True)
                    s = e
    length = torch.linalg.norm(normal, dim=1, keepdim=True)
    normal = torch.where(length > 1e-12, normal / length.clamp(min=1e-12), normals[tri])  # 法向互相抵消时退回最近三角形的法向
    return cp.cpu().numpy(), dist.cpu().numpy(), tri.cpu().numpy(), normal.cpu().numpy()
