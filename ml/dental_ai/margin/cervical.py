"""牙颈线（牙齿与牙龈的交界线）提取。

分割结果中每颗牙的区域边界就是天然牙的牙颈线。它有两个用途：
1. 展示和编辑（前端在网格上画出这条线）；
2. 作为生成牙底面的参考（相邻牙的牙颈线高度决定缺失牙的虚拟颈缘线）。
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import trimesh

from ..geometry import face_labels


def boundary_loops(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    """开放网格的边界环，按顶点顺序返回（每个环是顶点下标数组，首尾不重复）。"""
    edges = mesh.edges_sorted
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if len(boundary) == 0:
        return []
    graph = nx.Graph()
    graph.add_edges_from(boundary.tolist())
    loops = []
    for comp in nx.connected_components(graph):
        sub = graph.subgraph(comp)
        try:
            cycle = nx.find_cycle(sub)
        except nx.NetworkXNoCycle:
            continue
        loops.append(np.array([u for u, _ in cycle]))
    loops.sort(key=len, reverse=True)
    return loops


def cervical_line(mesh: trimesh.Trimesh, labels: np.ndarray, fdi: int, smooth_iters: int = 3) -> np.ndarray:
    """返回某颗牙的牙颈线折线坐标 (K, 3)，闭合环、按顺序排列。"""
    mask = face_labels(mesh, labels) == fdi
    if not mask.any():
        raise ValueError(f"网格中没有牙齿 {fdi}")
    tooth = mesh.submesh([np.nonzero(mask)[0]], append=True, repair=False)
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
