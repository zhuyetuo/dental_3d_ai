"""分割后处理：去除碎片、校正编号顺序。"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..fdi import Jaw, arch_order, expected_width
from ..geometry import estimate_arch_frame, tooth_centroids


def remove_small_islands(
    mesh: trimesh.Trimesh, labels: np.ndarray, min_vertices: int = 50, iters: int = 5, keep_fraction: float = 0.3
) -> np.ndarray:
    """同一标签的连通块中，小于 min_vertices、或小于最大块 keep_fraction 的小块按邻居多数票重新赋值。"""
    labels = labels.copy()
    edges = mesh.edges_unique
    n = len(mesh.vertices)
    # 没有被三角形引用的孤立顶点（真实 Teeth3DS 扫描里约 7%）自己就是一个“连通块”，又没有邻居可以投票，
    # 以前会被一律改成牙龈，牙齿少了 7% 的顶点，官方指标明显下降。这里跳过它们，最后跟随最近的被引用顶点
    referenced = np.zeros(n, dtype=bool)
    referenced[mesh.faces] = True
    for _ in range(iters):
        changed = False
        same = labels[edges[:, 0]] == labels[edges[:, 1]]
        e = edges[same]
        graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        _, comp = connected_components(graph, directed=False)
        sizes = np.bincount(comp)
        # 每个标签保留最大的连通块，以及不小于它 keep_fraction 的块。不能只留最大的一块：模型输出常被一条错分带
        # 把一颗牙切成两大块，第二块四周是牙龈，按邻居投票会整块变成牙龈（CPU 基线上 37 一次丢掉 1450 个顶点）
        keep = set()
        for lab in np.unique(labels):
            comps = np.unique(comp[labels == lab])
            largest = sizes[comps].max()
            keep.update(comps[sizes[comps] >= keep_fraction * largest].tolist())
        bad = ~np.isin(comp, list(keep)) | (sizes[comp] < min_vertices)
        bad &= (labels != 0) & referenced
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
    return follow_nearest_referenced(mesh, labels, referenced)


def follow_nearest_referenced(mesh: trimesh.Trimesh, labels: np.ndarray, referenced: np.ndarray | None = None) -> np.ndarray:
    """孤立顶点（没有三角形引用）的标签取最近的被引用顶点的标签。"""
    if referenced is None:
        referenced = np.zeros(len(mesh.vertices), dtype=bool)
        referenced[mesh.faces] = True
    isolated = np.nonzero(~referenced)[0]
    if len(isolated) == 0:
        return labels
    ref = np.nonzero(referenced)[0]
    _, nearest = cKDTree(np.asarray(mesh.vertices)[ref]).query(np.asarray(mesh.vertices)[isolated])
    labels = labels.copy()
    labels[isolated] = labels[ref[nearest]]
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


def _tooth_widths(labels: np.ndarray, points: np.ndarray):
    """每个牙号沿牙弓切线的宽度 / 该牙位的标准宽度（按整副牙列的比例归一），以及量宽度的函数和比例。"""
    fs = [int(f) for f in np.unique(labels) if f]
    cen = np.array([points[labels == f].mean(axis=0) for f in fs])
    coef = np.polyfit(cen[:, 0], cen[:, 1], 2)

    def width(mask):
        x = points[mask, 0].mean()
        t = np.array([1.0, 2 * coef[0] * x + coef[1]])
        t /= np.linalg.norm(t)
        p = points[mask][:, :2] @ t
        return float(np.percentile(p, 99) - np.percentile(p, 1))

    raw = {f: width(labels == f) for f in fs}
    scale = float(np.median([raw[f] / expected_width(f) for f in fs]))
    return {f: raw[f] / (expected_width(f) * scale) for f in fs}, width, scale


def split_tooth_candidates(mesh: trimesh.Trimesh, labels: np.ndarray, points: np.ndarray) -> list[tuple[int, int, float, float]]:
    """相邻的两个牙号是不是同一颗牙：[(小块牙号, 大块牙号, 接触占比, 合起来的宽度 / 一颗牙的标准宽度)]。

    接触占比 = 小块的边界顶点里直接挨着大块的比例。真正相邻的两颗牙只在邻接点附近相接（Teeth3DS 标注上中位数 0.20、
    99% 分位 0.34），一颗牙被切成两块时切口很长。宽度沿牙弓切线量，按整副牙列的比例归一。"""
    if len(np.unique(labels)) < 4:
        return []
    _, width, scale = _tooth_widths(labels, points)
    e = mesh.edges_unique
    la, lb = labels[e[:, 0]], labels[e[:, 1]]
    diff = la != lb
    size = {int(f): int(n) for f, n in zip(*np.unique(labels, return_counts=True))}
    both = diff & (la > 0) & (lb > 0)
    out = []
    for a, b in {tuple(sorted(p)) for p in zip(la[both].tolist(), lb[both].tolist())}:
        s, o = (a, b) if size[a] <= size[b] else (b, a)
        boundary = len(np.unique(np.concatenate([e[diff & (la == s), 0], e[diff & (lb == s), 1]])))
        m = both & (((la == s) & (lb == o)) | ((la == o) & (lb == s)))
        touch = len(np.unique(np.where(la[m] == s, e[m, 0], e[m, 1])))
        ratio = width((labels == s) | (labels == o)) / (max(expected_width(s), expected_width(o)) * scale)
        out.append((s, o, touch / max(boundary, 1), ratio))
    return out


def merge_split_teeth(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    jaw: Jaw,
    points: np.ndarray | None = None,
    min_contact: float = 0.4,
    max_width_ratio: float = 1.2,
    fused_ratio: float | None = 1.45,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """把“一颗牙被分成两个牙号”的两块合并成一颗，返回 (新标签, [(被并掉的牙号, 并入的牙号)])。

    缺牙且间隙已闭合的牙列上，分割模型倾向于凑满 14 颗牙：把缺牙旁边的一颗牙切成两个牙号，于是没有牙号空出来，
    缺牙检测不到。判据是两块之间的切口很长（接触占比 ≥ min_contact）而且合起来只有一颗牙宽（≤ max_width_ratio）。
    阈值在 Teeth3DS 测试集的模型输出上定的：满足判据的合并里没有把两颗真牙并成一颗的（真牙合起来的宽度都在 1.34 倍以上）。
    并入大块的牙号；到底缺的是哪一个牙号，几何上分不出来。

    fused_ratio：同一象限里另有一个牙号的宽度达到标准宽度的这么多倍时不合并。那通常是两颗牙被标成了一个号，
    模型在这里切开一颗牙是在“补”那个号，牙其实一颗不缺，合并了反而会误报缺牙。None 表示不做这个保护。
    points：标准坐标系下的顶点（见 orientation.py）；不提供时内部做一次朝向归一化。"""
    if points is None:
        from ..orientation import normalize_orientation

        points = normalize_orientation(mesh, jaw).apply(mesh).vertices
    points = np.asarray(points)
    labels = np.asarray(labels).copy()
    merged: list[tuple[int, int]] = []
    for _ in range(4):
        cand = [c for c in split_tooth_candidates(mesh, labels, points) if c[2] >= min_contact and c[3] <= max_width_ratio]
        if cand and fused_ratio is not None:
            ratios = _tooth_widths(labels, points)[0]
            cand = [c for c in cand if not any(f // 10 == c[1] // 10 and f not in c[:2] and r >= fused_ratio for f, r in ratios.items())]
        if not cand:
            break
        s, o, _, _ = max(cand, key=lambda c: c[2])
        labels[labels == s] = o
        merged.append((s, o))
    return labels, merged
