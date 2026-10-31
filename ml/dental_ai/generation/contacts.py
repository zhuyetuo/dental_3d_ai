"""邻接面调整：让生成的牙冠与近、远中邻牙的距离等于设计参数里的邻接距离。

对每颗邻牙迭代：
1. 牙冠表面每个可动顶点到邻牙表面的有符号距离（沿邻牙表面法向，邻牙外为正、穿入为负）；
2. 取最小值 d_min 和对应的接触点 c；与目标的差 Δ = d_min − target；
3. 接触点附近的顶点按高斯权重沿“朝向邻牙”的方向移动 Δ（Δ > 0 时靠近，< 0 时退开）；
4. 直到 |Δ| 足够小。

颈缘、底边附近的顶点不动（由调用方通过 movable 指定），所以不会破坏和底面、组织面的缝合。
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ..margin.cervical import boundary_loops


def signed_distance_to_surface(
    points: np.ndarray, surface: trimesh.Trimesh, edge_margin: float = 0.03, cutoff: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """到开放曲面（邻牙牙冠）的有符号距离（沿最近三角形法向，法向一侧为正）。返回 (距离, 最近三角形的法向)。

    最近点就落在曲面开放边缘上（edge_margin 以内）的点记为 +inf：这些点在曲面“范围之外”
    （如邻牙牙颈线以下、牙龈高度上的点），用边缘三角形的法向判断内外不可靠，会被误判为穿入。
    只排除最近点“在边上”的点，不排除最近点“靠近边”的点：真实口扫的邻牙标签边界在邻面处
    一直升到接触点，接触区就挨着开放边缘，排除范围大了（以前是 0.5 mm）就找不到接触。

    离邻牙顶点超过 cutoff 的点不算精确距离，直接用到最近顶点的距离（正数，法向记为 0）：
    真实口扫的牙冠有上万个顶点，全部求最近三角形要十几秒，而邻接只关心靠近邻牙的那一小片。"""
    d = cKDTree(surface.vertices).query(points)[0]
    normals = np.zeros_like(points, dtype=float)
    near = np.nonzero(d <= cutoff)[0]
    if len(near):
        cp, _, tri = trimesh.proximity.closest_point(surface, points[near])
        normals[near] = surface.face_normals[tri]
        d[near] = np.einsum("ij,ij->i", points[near] - cp, normals[near])
        loops = boundary_loops(surface)
        if loops and edge_margin > 0:
            # 边界折线加密，近似“到边界线段”的距离
            a = np.concatenate([surface.vertices[loop] for loop in loops])
            b = np.concatenate([surface.vertices[np.roll(loop, -1)] for loop in loops])
            dense = np.concatenate([a + (b - a) * t for t in np.linspace(0, 1, 8, endpoint=False)])
            on_edge = cKDTree(dense).query(cp)[0] < edge_margin
            d[near[on_edge]] = np.inf
    return d, normals


def adjust_contacts(
    crown: trimesh.Trimesh,
    neighbors: dict[int, trimesh.Trimesh],
    target: float,
    movable: np.ndarray,
    radius: float = 2.0,
    iterations: int = 12,
    tolerance: float = 0.01,
    max_gap: float = 3.0,
) -> tuple[dict[int, float], list[str]]:
    """原地修改 crown.vertices。返回 ({邻牙: 调整后的最小有符号距离}, 警告)。"""
    result, warnings = {}, []
    idx = np.nonzero(movable)[0]
    if len(idx) == 0:
        return result, ["没有可调整的顶点，跳过邻接调整"]
    sigma2 = (radius / 2) ** 2
    for fdi, surface in neighbors.items():
        d_min = np.inf
        for _ in range(iterations):
            d, normals = signed_distance_to_surface(crown.vertices[idx], surface)
            k = int(np.argmin(d))
            d_min = float(d[k])
            if not np.isfinite(d_min):
                break
            if d_min > max_gap:
                warnings.append(f"与 {fdi} 的间隙 {d_min:.1f} mm 太大（可能是牙间隙），未调整邻接")
                break
            delta = d_min - target
            if abs(delta) <= tolerance:
                break
            contact = crown.vertices[idx[k]]
            toward = -normals[k]  # 邻牙法向朝外，朝向邻牙就是反方向
            w = np.exp(-((crown.vertices[idx] - contact) ** 2).sum(axis=1) / (2 * sigma2))
            crown.vertices[idx] = crown.vertices[idx] + (w * delta)[:, None] * toward
        result[fdi] = round(d_min, 3) if np.isfinite(d_min) else None
        if np.isfinite(d_min) and d_min <= max_gap and abs(d_min - target) > 0.05:
            warnings.append(f"与 {fdi} 的邻接距离 {d_min:.2f} mm，未达到目标 {target:.2f} mm")
    return result, warnings
