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


def signed_distance_to_surface(points: np.ndarray, surface: trimesh.Trimesh, edge_margin: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """到开放曲面（邻牙牙冠）的有符号距离（沿最近三角形法向，法向一侧为正）。返回 (距离, 最近三角形的法向)。

    最近点落在曲面开放边缘 edge_margin 以内的点记为 +inf：邻牙在牙颈线以下没有面，
    靠近边缘时用边缘三角形的法向判断内外不可靠，会把牙龈高度上并未接触的点误判为穿入。"""
    cp, _, tri = trimesh.proximity.closest_point(surface, points)
    normals = surface.face_normals[tri]
    d = np.einsum("ij,ij->i", points - cp, normals)
    loops = boundary_loops(surface)
    if loops and edge_margin > 0:
        edge = surface.vertices[np.concatenate(loops)]
        near_edge = cKDTree(edge).query(cp)[0] < edge_margin
        d = np.where(near_edge, np.inf, d)
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
