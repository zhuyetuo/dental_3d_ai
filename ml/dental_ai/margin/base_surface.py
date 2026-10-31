"""为生成的牙冠构造贴合牙龈的底面，输出封闭（watertight）网格。

缺失牙（种植冠 / 桥体）没有预备体颈缘线，底面要贴合缺牙区牙龈：
0. 先把牙冠底边切成绕牙轴单调的扇贝形（真实牙颈线起伏大，见 trim_to_scallop）；
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
    """沿 -up 方向把点投影到曲面上。射线没有命中的点（如伸出扫描边缘）保持水平位置，高度取最近点的高度：
    直接用最近点的话，几个点会落到同一个边缘顶点上，导出 STL 时重合点被焊接，网格就不再是流形。"""
    origins = points + up * reach
    directions = np.tile(-up, (len(points), 1))
    nearest = surface.nearest.on_surface(points)[0]
    out = points + ((nearest - points) @ up)[:, None] * up
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


def _polar(points: np.ndarray, center: np.ndarray, up: np.ndarray, u: np.ndarray, v: np.ndarray):
    d = points - center
    return np.arctan2(d @ v, d @ u), d @ up


def trim_to_scallop(crown: trimesh.Trimesh, up: np.ndarray, bins: int = 72, clearance: float = 0.2) -> trimesh.Trimesh:
    """沿一条光滑的扇贝形曲线切掉牙冠底边，使开放边界绕牙轴单调（每个方位只有一个高度）。

    真实牙齿的牙颈线是起伏的空间曲线（邻面处向咬合方向升高 2–4 mm），水平投影不是星形，
    甚至自相交；直接向内收拢再投影到牙龈，点会挤在一起，封底后网格不再是流形。
    切割高度 = 各方位上边界最高点的上包络（先取邻域最大值再平均，保证不低于原边界）+ clearance。"""
    from ..geometry import clean_crown
    from .cut import split_along_zero

    loop = boundary_loops(crown)[0]
    boundary = crown.vertices[loop]
    center = boundary.mean(axis=0)
    u = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(up, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(up, u)

    theta_b, h_b = _polar(boundary, center, up, u, v)
    k = np.floor((theta_b + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    top = np.full(bins, -np.inf)
    np.maximum.at(top, k, h_b)
    idx = np.arange(bins)
    have = np.isfinite(top)
    top = np.interp(idx, idx[have], top[have], period=bins)
    dilated = np.max([np.roll(top, s) for s in range(-2, 3)], axis=0)
    envelope = np.mean([np.roll(dilated, s) for s in range(-2, 3)], axis=0) + clearance

    theta, h = _polar(crown.vertices, center, up, u, v)
    x = (theta + np.pi) / (2 * np.pi) * bins - 0.5  # 以格子中心为插值节点
    f = np.interp(x, idx, envelope, period=bins) - h  # > 0：切割线以下，切掉
    cut, values = split_along_zero(crown, f)
    keep = (values <= 0)[cut.faces].all(axis=1)
    if not keep.any():
        raise ValueError("牙冠太矮，切平底边后没有剩余")
    trimmed = trimesh.Trimesh(cut.vertices, cut.faces[keep], process=False)
    trimmed.remove_unreferenced_vertices()
    return clean_crown(trimmed)


def close_with_gum_base(
    crown: trimesh.Trimesh,
    gum: trimesh.Trimesh,
    up: np.ndarray,
    relief: float = 0.05,
    taper: float = 0.85,
    rings: int = 4,
    samples: int = 96,
) -> trimesh.Trimesh:
    """把开放的牙冠网格封底，使底面贴合牙龈。

    crown: 只有一个开放边界环的牙冠网格（法向朝外）
    gum:   牙龈网格（分割结果中标签为 0 的部分）
    up:    咬合方向单位向量

    先把底边切成绕牙轴单调的扇贝形（trim_to_scallop），再按方位角均匀取 samples 个方向构造
    虚拟颈缘线和底面同心环；牙冠边界与虚拟颈缘线之间按角度交替连接。每条边界边恰好接一个三角形，
    拓扑上一定封闭；不合并顶点（合并重合点会产生非流形边）。
    """
    from ..generation.prep_crown import _frame, _interp_loop, _loft, _loop_by_angle

    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)
    if len(boundary_loops(crown)) != 1:
        raise ValueError(f"牙冠应只有一个开放边界，实际有 {len(boundary_loops(crown))} 个")
    crown = trim_to_scallop(crown, up)
    loop = boundary_loops(crown)[0]
    boundary = crown.vertices[loop]
    frame = _frame(boundary, up)
    center = frame.center

    def horizontal_scale(pts: np.ndarray, s: float) -> np.ndarray:
        offset = pts - center
        vertical = (offset @ up)[:, None] * up
        return center + vertical + (offset - vertical) * s

    def on_gum(pts: np.ndarray) -> np.ndarray:
        return _project_to_surface(pts, gum, up) + up * relief

    order, ang = _loop_by_angle(boundary, frame)
    q = ang[0] + np.linspace(0, 2 * np.pi, samples, endpoint=False)
    profile = _interp_loop(boundary, frame, q)

    vertices = [crown.vertices]
    faces = [crown.faces]
    n = len(crown.vertices)

    # 虚拟颈缘线
    margin = on_gum(horizontal_scale(profile, taper))
    margin_idx = np.arange(n, n + samples)
    vertices.append(margin)
    faces.append(_loft(loop[order], ang, margin_idx, q))
    n += samples

    # 底面同心环
    prev = margin_idx
    for r in range(1, rings):
        ring = on_gum(horizontal_scale(profile, taper * (1 - r / rings)))
        idx = np.arange(n, n + samples)
        vertices.append(ring)
        faces.append(_strip(prev, idx))
        prev = idx
        n += samples

    tip = on_gum(center[None, :])
    vertices.append(tip)
    i = np.arange(samples)
    faces.append(np.stack([prev[(i + 1) % samples], prev[i], np.full(samples, n)], axis=1))

    result = trimesh.Trimesh(np.vstack(vertices), np.vstack(faces), process=False)
    # 新加的三角形绕序按构造方式不一定与牙冠一致：统一绕序，并让法向朝外
    trimesh.repair.fix_normals(result)
    return result
