"""为生成的牙冠构造贴合牙龈的底面，输出封闭（watertight）网格。

缺失牙（种植冠 / 桥体）没有预备体颈缘线，底面要贴合缺牙区牙龈：
1. 牙冠开放边界沿咬合法向反方向投影到牙龈上，并向内收拢（形成穿龈轮廓），得到“虚拟颈缘线”；
2. 牙冠边界与虚拟颈缘线之间生成侧壁；
3. 虚拟颈缘线内部按同心环采样，每个点再投影到牙龈表面，得到贴合牙龈的底面；
4. relief 控制底面与牙龈的距离：>0 留间隙，<0 压入牙龈（卵圆形桥体）。

预备体牙冠（有颈缘线）走另一条路：内表面 = 预备体表面外扩粘接间隙，见 docs/architecture.md。
"""

from __future__ import annotations

import numpy as np
import trimesh

from .cervical import boundary_loops


def _project_to_surface(points: np.ndarray, surface: trimesh.Trimesh, up: np.ndarray, reach: float = 30.0) -> np.ndarray:
    """沿 -up 方向把点投影到曲面上；射线没有命中的点退化为最近点。"""
    origins = points + up * reach
    directions = np.tile(-up, (len(points), 1))
    out = surface.nearest.on_surface(points)[0]
    locations, ray_idx, _ = surface.ray.intersects_location(origins, directions, multiple_hits=False)
    out[ray_idx] = locations
    return out


def _strip(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """两个等长闭合环之间的三角带。outer 的 i->i+1 方向边已被已有三角形使用。"""
    k = len(outer)
    i = np.arange(k)
    j = (i + 1) % k
    t1 = np.stack([outer[j], outer[i], inner[i]], axis=1)
    t2 = np.stack([outer[j], inner[i], inner[j]], axis=1)
    return np.concatenate([t1, t2])


def _loop_matches_face_winding(mesh: trimesh.Trimesh, loop: np.ndarray) -> bool:
    a, b = loop[0], loop[1]
    for face in mesh.faces[np.any(mesh.faces == a, axis=1)]:
        for s in range(3):
            if face[s] == a and face[(s + 1) % 3] == b:
                return True
    return False


def close_with_gum_base(
    crown: trimesh.Trimesh,
    gum: trimesh.Trimesh,
    up: np.ndarray,
    relief: float = 0.05,
    taper: float = 0.85,
    rings: int = 4,
) -> trimesh.Trimesh:
    """把开放的牙冠网格封底，使底面贴合牙龈。

    crown: 只有一个开放边界环的牙冠网格（法向朝外）
    gum:   牙龈网格（分割结果中标签为 0 的部分）
    up:    咬合方向单位向量
    """
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)
    loops = boundary_loops(crown)
    if len(loops) != 1:
        raise ValueError(f"牙冠应只有一个开放边界，实际有 {len(loops)} 个")
    loop = loops[0]
    if not _loop_matches_face_winding(crown, loop):
        loop = loop[::-1]

    boundary = crown.vertices[loop]
    center = boundary.mean(axis=0)

    def horizontal_scale(pts: np.ndarray, s: float) -> np.ndarray:
        offset = pts - center
        vertical = (offset @ up)[:, None] * up
        return center + vertical + (offset - vertical) * s

    def on_gum(pts: np.ndarray) -> np.ndarray:
        return _project_to_surface(pts, gum, up) + up * relief

    vertices = [crown.vertices]
    faces = [crown.faces]
    n = len(crown.vertices)

    # 虚拟颈缘线
    margin = on_gum(horizontal_scale(boundary, taper))
    margin_idx = np.arange(n, n + len(margin))
    vertices.append(margin)
    faces.append(_strip(loop, margin_idx))
    n += len(margin)

    # 底面同心环
    prev = margin_idx
    for r in range(1, rings):
        ring = on_gum(horizontal_scale(boundary, taper * (1 - r / rings)))
        idx = np.arange(n, n + len(ring))
        vertices.append(ring)
        faces.append(_strip(prev, idx))
        prev = idx
        n += len(ring)

    tip = on_gum(center[None, :])
    vertices.append(tip)
    k = len(prev)
    i = np.arange(k)
    faces.append(np.stack([prev[(i + 1) % k], prev[i], np.full(k, n)], axis=1))

    result = trimesh.Trimesh(np.vstack(vertices), np.vstack(faces), process=False)
    result.merge_vertices()
    return result
