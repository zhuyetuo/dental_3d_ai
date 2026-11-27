"""窝沟锐化：在现有的大窝沟里找中心线，沿中心线刻一条窄的 V 形沟。
技师牙冠上的窝沟又窄又利；泊松重建 + 加深出来的沟位置基本对（IoU 0.42），但又宽又软。这里不改沟的位置，只把它变窄变利。
40 例真实预备体（16 号）：细沟幅度 0.038 → 0.051 mm（技师 0.055）、粗糙度 3.6° → 6.2°（技师 7.9°）、窝沟位置 IoU 不变、CD-L2 +0.007。
坐标：数据坐标（y 朝咬合面），和 voxel_crown / occlusal 一样。"""
from __future__ import annotations

import os

import numpy as np

K8 = np.ones((3, 3))


def _prune(sk, iters):
    """剪毛刺：反复去掉端点（8 邻域里只有 1 个邻居的格子）。"""
    from scipy import ndimage

    sk = sk.copy()
    for _ in range(iters):
        nb = ndimage.convolve(sk.astype(int), K8, mode="constant") - sk
        end = sk & (nb <= 1)
        if not end.any():
            break
        sk &= ~end
    return sk


def centerlines(g, valid, cell, thr=0.06, min_len_mm=1.0, sigma_cells=2.0, prune_mm=0.35, edge_mm=0.9):
    """沟深图的脊线 = 沟的中心线（在某个方向上比两边都深的格子），剪毛刺、去短段、离咬合面轮廓 edge_mm 以内不要。"""
    from scipy import ndimage

    gs = ndimage.gaussian_filter(g, sigma_cells)
    P = np.pad(gs, 1, mode="edge")
    c = P[1:-1, 1:-1]
    ridge = np.zeros_like(c, bool)
    for (a0, a1), (b0, b1) in (((0, 1), (2, 1)), ((1, 0), (1, 2)), ((0, 0), (2, 2)), ((0, 2), (2, 0))):
        A = P[a0:a0 + c.shape[0], a1:a1 + c.shape[1]]
        B = P[b0:b0 + c.shape[0], b1:b1 + c.shape[1]]
        ridge |= (c > A) & (c >= B)
    inner = ndimage.distance_transform_edt(valid) * cell
    sk = ridge & (gs > thr) & valid & (inner > edge_mm)
    sk = _prune(sk, int(round(prune_mm / cell)))
    lab, n = ndimage.label(sk, structure=K8)
    if n:
        sizes = ndimage.sum(sk, lab, range(1, n + 1))
        sk &= np.isin(lab, 1 + np.flatnonzero(sizes * cell >= min_len_mm))
    return sk


def carve_map(h, valid, cell, groove_fn, half_width=0.15, depth=0.12, rel=0.6):
    """高度图上的负位移（mm）：V 形，中心深 depth + rel × 当地沟深，半宽 half_width。"""
    from scipy import ndimage

    g = groove_fn(h, valid)
    sk = centerlines(g, valid, cell)
    if not sk.any():
        return np.zeros_like(h)
    dist, idx = ndimage.distance_transform_edt(~sk, return_indices=True)
    prof = np.clip(1 - dist * cell / half_width, 0, 1) ** 1.5
    g_at = ndimage.gaussian_filter(g, 2.0)[tuple(idx)]
    inner = ndimage.distance_transform_edt(valid) * cell
    fade = np.clip((inner - 0.9) / 0.6, 0, 1)
    return ndimage.gaussian_filter(-(depth + rel * g_at) * prof * fade * valid, 0.8)


def sharpen(verts, faces, subdivide: bool = True):
    """牙冠外表面（数据坐标）→ 刻好窄沟的网格 (verts, faces)。VOXEL_SHARPEN="半宽,深度,比例"，"0" 关掉。"""
    import trimesh
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from trimesh.grouping import group_rows

    from . import occlusal as oc

    spec = os.environ.get("VOXEL_SHARPEN", "0.15,0.12,0.6")
    if spec.strip() in ("", "0", "off"):
        return verts, faces
    hw, dep, rel = (float(x) for x in spec.split(","))
    m0 = trimesh.Trimesh(np.asarray(verts, float), np.asarray(faces), process=False)
    pts = np.asarray(trimesh.sample.sample_surface(m0, 300000, seed=1)[0])
    x4, (cx, cz, top) = oc.inputs(pts, None)
    mc = x4[1] > 0
    dmap = carve_map(x4[0] + top, mc, oc.CELL, oc.groove_depth_map, hw, dep, rel)
    if not np.any(dmap):
        return verts, faces
    v, f = np.asarray(verts, float), np.asarray(faces)
    if subdivide:  # 窄沟要细一点的网格才刻得出来
        v, f = trimesh.remesh.subdivide(v, f)
    m = trimesh.Trimesh(v, f, process=False)
    n = m.vertex_normals
    fx = (v[:, 0] - cx) / oc.CELL + oc.RES / 2 - 0.5
    fz = (v[:, 2] - cz) / oc.CELL + oc.RES / 2 - 0.5
    dv = ndimage.map_coordinates(dmap, [fx, fz], order=1, mode="constant", cval=0.0)
    w = np.clip((n[:, 1] - 0.3) / 0.3, 0, 1) * (v[:, 1] >= top - oc.TOP_BAND)
    es = m.edges_sorted
    rim = np.unique(es[group_rows(es, require_count=1)])
    if len(rim):
        w *= np.clip(cKDTree(v[rim]).query(v)[0] / 1.0, 0, 1)
    v = v.copy()
    v[:, 1] += dv * w
    return v, f
