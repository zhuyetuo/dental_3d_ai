"""分割后处理：去除碎片、校正编号顺序。"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..fdi import Jaw, arch_order
from ..geometry import estimate_arch_frame, tooth_centroids


def remove_small_islands(mesh: trimesh.Trimesh, labels: np.ndarray, min_vertices: int = 50, iters: int = 5) -> np.ndarray:
    """同一标签的每个连通块中，只保留最大的一块；其它小块按邻居多数票重新赋值。"""
    labels = labels.copy()
    edges = mesh.edges_unique
    n = len(mesh.vertices)
    for _ in range(iters):
        changed = False
        same = labels[edges[:, 0]] == labels[edges[:, 1]]
        e = edges[same]
        graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        _, comp = connected_components(graph, directed=False)
        sizes = np.bincount(comp)
        # 每个标签最大的连通块
        keep = set()
        for lab in np.unique(labels):
            comps = np.unique(comp[labels == lab])
            keep.add(comps[np.argmax(sizes[comps])])
        bad = ~np.isin(comp, list(keep)) | (sizes[comp] < min_vertices)
        bad &= labels != 0
        if not bad.any():
            break
        neighbors = mesh.vertex_neighbors
        for v in np.nonzero(bad)[0]:
            votes = labels[neighbors[v]]
            votes = votes[votes != labels[v]]
            if len(votes):
                labels[v] = np.bincount(votes).argmax()
                changed = True
            else:
                labels[v] = 0
                changed = True
        if not changed:
            break
    return labels


def check_arch_order(mesh: trimesh.Trimesh, labels: np.ndarray, jaw: Jaw) -> list[str]:
    """检查牙号是否沿牙弓单调排列，返回问题描述列表（空表示正常）。

    把每颗牙质心投影到牙弓左右方向，按 FDI 期望顺序排序后，横坐标应大致单调递增
    （中切牙附近牙弓弯曲，允许少量回折，这里只检查同一象限内部）。
    """
    frame = estimate_arch_frame(mesh, labels, jaw)
    cents = tooth_centroids(mesh, labels)
    order = [f for f in arch_order(jaw, include_third_molars=True) if f in cents]
    issues = []
    x = {f: float((cents[f] - frame.origin) @ frame.lateral) for f in order}
    # 顺序按“投影到牙弓曲线上的位置”比较，不直接比左右坐标：拥挤、颊侧错位的牙（常见是尖牙）
    # 左右坐标会颠倒，但沿牙弓的顺序是对的（Poseidon3D 正畸病例上的正确标注也会被误报）
    posterior = np.cross(frame.occlusal, frame.lateral)
    pts = np.array([[x[f], float((cents[f] - frame.origin) @ posterior)] for f in order])
    along = np.array([x[f] for f in order])
    if len(pts) >= 3:
        coef = np.polyfit(pts[:, 0], pts[:, 1], 2)
        xs = np.linspace(pts[:, 0].min() - 10, pts[:, 0].max() + 10, 4001)
        _, idx = cKDTree(np.stack([xs, np.polyval(coef, xs)], axis=1)).query(pts)
        along = xs[idx]
    pos = dict(zip(order, along))
    for a, b in zip(order, order[1:]):
        if a // 10 == b // 10 and pos[b] < pos[a]:
            issues.append(f"{a} 与 {b} 的左右顺序颠倒，可能编号错误")
    for f in order:
        side = x[f]
        right = f // 10 in (1, 4)
        if (right and side > 0) or (not right and side < 0):
            issues.append(f"{f} 位于中线的另一侧，可能左右标反")
    return issues
