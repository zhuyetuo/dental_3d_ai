"""沿颈缘线精确切开网格，得到边界落在颈缘线上的预备体表面。

做法：给每个顶点一个有符号值 f（颈缘线内侧 < 0，外侧 > 0），把网格沿 f = 0 切开：

    f = 到颈缘线水平投影多边形的有符号距离（多边形内为负）
        + max(0, 颈缘线在该方位的高度 − 顶点高度)

零值面就是“在颈缘线投影以内、且不低于颈缘线”这个区域的边界：
- 肩台面上（投影在内、高度约等于颈缘线）f < 0；颈缘线外的牙龈 f > 0；
- 竖直的台阶面上，水平距离约为 0，零值正好落在颈缘线的高度，也就是肩台外缘的尖角处。

试过用牙面法向定义“外侧方向”的版本，在尖角附近法向不稳定，会把颈缘外的牙龈切进来，已放弃。
切开时只拆分跨过 f = 0 的三角形，新顶点都在原来的边上，所以切割线一定在原始牙面上。
"""

from __future__ import annotations

import numpy as np
import trimesh


def polygon_signed_distance(points: np.ndarray, poly: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """二维点到闭合多边形边界的距离，多边形内为负。"""
    a = poly
    b = np.roll(poly, -1, axis=0)
    ab = b - a
    best = np.full(len(points), np.inf)
    for k in range(len(poly)):
        ap = points - a[k]
        t = np.clip((ap @ ab[k]) / max(ab[k] @ ab[k], 1e-12), 0.0, 1.0)
        d = np.linalg.norm(ap - t[:, None] * ab[k], axis=1)
        best = np.minimum(best, d)
    return np.where(inside, -best, best)


def margin_field(xy: np.ndarray, h: np.ndarray, poly: np.ndarray, inside: np.ndarray, margin_height: np.ndarray) -> np.ndarray:
    """xy、h：顶点在牙轴坐标系下的水平坐标和高度；margin_height：每个顶点所在方位的颈缘线高度。"""
    return polygon_signed_distance(xy, poly, inside) + np.maximum(0.0, margin_height - h)


def split_along_zero(mesh: trimesh.Trimesh, f: np.ndarray, snap: float = 0.03) -> tuple[trimesh.Trimesh, np.ndarray]:
    """沿 f = 0 切开三角形。返回 (新网格, 新顶点的 f 值；切割线上的顶点为 0)。

    snap：离切割线这么近（|f| < snap）的顶点统一归到内侧（f = −snap）。否则切割点会紧贴着顶点，
    切出面积几乎为 0 的三角形，网格就不再封闭。代价是切割线最多向外偏 snap。"""
    f = np.where(np.abs(f) < snap, -snap, f)
    verts = [mesh.vertices]
    values = [f]
    edge_point: dict[tuple[int, int], int] = {}
    n = len(mesh.vertices)

    def cut(a: int, b: int) -> int:
        nonlocal n
        key = (a, b) if a < b else (b, a)
        if key not in edge_point:
            t = f[a] / (f[a] - f[b])
            verts.append((mesh.vertices[a] + t * (mesh.vertices[b] - mesh.vertices[a]))[None])
            values.append(np.zeros(1))
            edge_point[key] = n
            n += 1
        return edge_point[key]

    faces = []
    neg = f < 0
    for tri in mesh.faces:
        s = neg[tri]
        if s.all() or not s.any():
            faces.append(tri)
            continue
        # 旋转顶点，让“单独在一侧”的顶点排在第一个，保持绕序
        lone = int(np.nonzero(s if s.sum() == 1 else ~s)[0][0])
        a, b, c = tri[lone], tri[(lone + 1) % 3], tri[(lone + 2) % 3]
        p, q = cut(a, b), cut(a, c)
        faces += [[a, p, q], [p, b, c], [p, c, q]]
    out = trimesh.Trimesh(np.vstack(verts), np.array(faces), process=False)
    return out, np.concatenate(values)
