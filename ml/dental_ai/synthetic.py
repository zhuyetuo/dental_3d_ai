"""合成牙弓：用于测试和前后端联调，在拿到真实口扫数据之前也能跑通整个流程。

网格是一张高度场：牙龈是平缓的马蹄形隆起，每颗牙是牙弓抛物线上的一个凸包，
牙齿区域与牙龈区域共享边，因此每颗牙都有真实的开放边界（牙颈线）。
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .fdi import Jaw, arch_order, expected_width


def synthetic_arch(
    jaw: Jaw = "lower",
    missing: tuple[int, ...] = (),
    resolution: float = 0.35,
    curvature: float = 0.045,
    band: float | None = None,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """返回 (网格, 逐顶点 FDI 标签)。

    坐标：单位 mm，牙冠朝 +Z，+Y 指向后方（磨牙），+X 为患者左侧。下颌与 Teeth3DS 标准坐标系一致；
    Teeth3DS 中上颌牙冠朝 -Z，需要时把上颌关于 XY 平面镜像（见 tests 中的 canonical_upper）。
    band：只保留离牙弓曲线 band mm 以内的部分，得到真实口扫那样的马蹄形轮廓；None 表示保留整块矩形。
    """
    order = arch_order(jaw)
    widths = np.array([expected_width(f) for f in order])
    # 沿抛物线 y = curvature * x^2 按弧长排牙，中线位于两颗中切牙之间
    xs = np.linspace(-40, 40, 4001)
    ys = curvature * xs**2
    arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    arc -= np.interp(0, xs, arc)
    edges = np.concatenate([[0], np.cumsum(widths)])
    edges -= edges[len(order) // 2]
    centers_s = (edges[:-1] + edges[1:]) / 2
    cx = np.interp(centers_s, arc, xs)
    cy = curvature * cx**2

    gx = np.arange(cx.min() - 8, cx.max() + 8, resolution)
    gy = np.arange(-8, cy.max() + 8, resolution)
    X, Y = np.meshgrid(gx, gy)

    # 点到牙弓曲线的距离（离散近似）
    curve = cKDTree(np.stack([xs, ys], axis=1))
    d_curve = curve.query(np.stack([X.ravel(), Y.ravel()], axis=1))[0].reshape(X.shape)
    Z = 4.0 * np.exp(-(d_curve**2) / (2 * 4.0**2))  # 牙龈嵴

    labels = np.zeros(X.shape, dtype=np.int64)
    best = np.zeros(X.shape)
    for f, x0, y0, w in zip(order, cx, cy, widths):
        if f in missing:
            continue
        r = w * 0.45
        bump = np.clip(1 - ((X - x0) ** 2 + (Y - y0) ** 2) / r**2, 0, None)
        h = 7.0 * np.sqrt(bump)
        inside = (h > 0) & (h > best)
        best = np.where(inside, h, best)
        labels[inside] = f
    Z = Z + best

    ny, nx = X.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    mesh = trimesh.Trimesh(verts, faces, process=False)
    labels = labels.ravel()
    if band is not None:
        keep = (d_curve.ravel() <= band)[faces].all(axis=1)
        used = np.unique(faces[keep])
        remap = np.full(len(verts), -1)
        remap[used] = np.arange(len(used))
        mesh = trimesh.Trimesh(verts[used], remap[faces[keep]], process=False)
        labels = labels[used]
    return mesh, labels
