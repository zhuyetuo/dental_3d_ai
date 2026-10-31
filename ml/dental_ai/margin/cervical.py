"""牙颈线（牙齿与牙龈的交界线）提取。

分割结果中每颗牙的区域边界就是天然牙的牙颈线。它有两个用途：
1. 展示和编辑（前端在网格上画出这条线）；
2. 作为生成牙底面的参考（相邻牙的牙颈线高度决定缺失牙的虚拟颈缘线）。
"""

from __future__ import annotations

import numpy as np
import trimesh

from ..geometry import clean_crown, face_labels


def boundary_loops(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    """开放网格的边界环（每个环是顶点下标数组，首尾不重复），按周长从长到短排列。

    沿三角形绕序把边界边连起来走。真实口扫的边界常有“夹点”：同一个顶点上边界自己碰到自己
    （或碰到旁边的小洞）。这里在夹点处把它拆成几个简单环，最长的那个就是牙颈线 / 扫描外缘。
    （以前用无向图找任意一个环，在夹点附近会只找到一个 1–4 mm 的小环。）"""
    faces = mesh.faces
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, inv, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    boundary = directed[counts[inv.reshape(-1)] == 1]
    if len(boundary) == 0:
        return []
    out: dict[int, list[int]] = {}
    for u, v in boundary.tolist():
        out.setdefault(u, []).append(v)

    loops = []
    for start in list(out):
        while out[start]:
            path, pos, u = [start], {start: 0}, start
            while out.get(u):
                v = out[u].pop()
                if v in pos:  # 回到走过的顶点：切下一个简单环，继续走剩下的
                    i = pos[v]
                    cycle = path[i:]
                    if len(cycle) >= 3:
                        loops.append(np.array(cycle))
                    for w in path[i + 1 :]:
                        del pos[w]
                    path = path[: i + 1]
                    if len(path) == 1 and not out.get(v):
                        break
                else:
                    pos[v] = len(path)
                    path.append(v)
                u = v
            # 绕序不一致的网格可能走进死胡同，剩下的半截路径丢弃

    def perimeter(loop: np.ndarray) -> float:
        p = mesh.vertices[loop]
        return float(np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1).sum())

    loops.sort(key=perimeter, reverse=True)
    return loops


def cervical_line(mesh: trimesh.Trimesh, labels: np.ndarray, fdi: int, smooth_iters: int = 3) -> np.ndarray:
    """返回某颗牙的牙颈线折线坐标 (K, 3)，闭合环、按顺序排列。"""
    mask = face_labels(mesh, labels) == fdi
    if not mask.any():
        raise ValueError(f"网格中没有牙齿 {fdi}")
    tooth = clean_crown(mesh.submesh([np.nonzero(mask)[0]], append=True, repair=False))
    loops = boundary_loops(tooth)
    if not loops:
        raise ValueError(f"牙齿 {fdi} 没有开放边界，无法提取牙颈线")
    pts = tooth.vertices[loops[0]].copy()
    # 网格边界呈锯齿状，做几次闭合环上的拉普拉斯平滑
    for _ in range(smooth_iters):
        pts = 0.5 * pts + 0.25 * (np.roll(pts, 1, axis=0) + np.roll(pts, -1, axis=0))
    return pts


def all_cervical_lines(mesh: trimesh.Trimesh, labels: np.ndarray) -> dict[int, list[list[float]]]:
    lines = {}
    for fdi in np.unique(labels):
        if fdi == 0:
            continue
        try:
            lines[int(fdi)] = cervical_line(mesh, labels, int(fdi)).round(4).tolist()
        except ValueError:
            continue
    return lines
