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
        # 大小用到最近点的真实距离，法向只决定正负。最近点落在棱或边上时，点到最近点的连线并不沿法向，
        # 只取法向分量会把“隔着 0.9 mm、在侧面”的点算成“只差 0.03 mm”
        offset = points[near] - cp
        along = np.einsum("ij,ij->i", offset, normals[near])
        sign = np.where(along < 0, -1.0, 1.0)
        length = np.linalg.norm(offset, axis=1)
        d[near] = sign * length
        # 返回的方向也改成“最近点 → 点”的方向（穿入时取反），和距离一致；点正对着三角形时它就是法向。
        # 邻接调整沿这个方向移动顶点：最近点在棱上时沿法向移动并不会靠近邻牙
        ok = length > 1e-9
        normals[near[ok]] = offset[ok] / length[ok, None] * sign[ok, None]
        loops = boundary_loops(surface)
        if loops and edge_margin > 0:
            # 边界折线加密，近似“到边界线段”的距离
            a = np.concatenate([surface.vertices[loop] for loop in loops])
            b = np.concatenate([surface.vertices[np.roll(loop, -1)] for loop in loops])
            dense = np.concatenate([a + (b - a) * t for t in np.linspace(0, 1, 8, endpoint=False)])
            on_edge = cKDTree(dense).query(cp)[0] < edge_margin
            d[near[on_edge]] = np.inf
    return d, normals


class _Overlap:
    """核对“穿入邻牙”是不是真的：沿牙冠 → 邻牙的方向逐条线看，顶点有没有越过邻牙朝向牙冠一侧的表面。

    真实口扫上邻牙是按标签切出来的开放曲面，边界就在邻接点旁边。牙冠上靠近这条边界的顶点，最近的三角形常常是
    邻牙侧壁的背面，按法向算出来是“穿入零点几毫米”，其实还隔着一条缝（Bite2Text 上量到过：报告 −0.03 mm 已到位，
    实际还差 0.9 mm）。越过了才算穿入；没越过的按“在外面”处理，距离取绝对值。"""

    def __init__(self, crown_points: np.ndarray, surface: trimesh.Trimesh, cell: float = 0.3):
        other = np.asarray(trimesh.sample.sample_surface(surface, 15000, seed=0)[0])
        axis = other.mean(axis=0) - crown_points.mean(axis=0)
        self.axis = axis / np.linalg.norm(axis)
        a = np.cross(self.axis, [0.0, 0.0, 1.0])
        if np.linalg.norm(a) < 0.3:
            a = np.cross(self.axis, [0.0, 1.0, 0.0])
        self.a = a / np.linalg.norm(a)
        self.b = np.cross(self.axis, self.a)
        self.cell = cell
        s = other @ self.axis
        beyond = s > float(np.median(crown_points @ self.axis))  # 只要邻牙在牙冠中心外侧的那些点
        self.near: dict[tuple[int, int], float] = {}
        for key, value in zip(self._cells(other[beyond]), s[beyond]):
            if value < self.near.get(key, np.inf):
                self.near[key] = float(value)

    def _cells(self, points: np.ndarray) -> list[tuple[int, int]]:
        ia = np.floor(points @ self.a / self.cell).astype(int)
        ib = np.floor(points @ self.b / self.cell).astype(int)
        return list(zip(ia.tolist(), ib.tolist()))

    def crossed(self, points: np.ndarray, tolerance: float = 0.02) -> np.ndarray:
        s = points @ self.axis
        return np.array([s_i > self.near.get(key, np.inf) - tolerance for key, s_i in zip(self._cells(points), s)], dtype=bool)


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
        check = _Overlap(crown.vertices[idx], surface)
        for _ in range(iterations):
            d, normals = signed_distance_to_surface(crown.vertices[idx], surface)
            fake = np.flatnonzero(np.isfinite(d) & (d < 0))
            if len(fake):
                fake = fake[~check.crossed(crown.vertices[idx[fake]])]
                d[fake] = -d[fake]  # 在邻牙侧壁的背面、但并没有越过它：其实在外面
                normals[fake] = -normals[fake]
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
