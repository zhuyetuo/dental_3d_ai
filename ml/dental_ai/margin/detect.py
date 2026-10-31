"""预备体颈缘线自动检测（几何方法，不需要训练数据）。

预备体的颈缘线是牙体预备留下的终止线：肩台、斜面或刃状边缘的外缘，在网格上表现为一圈**凸起的棱**，
也就是相邻三角形法向夹角最大、且是凸角的地方。分割得到的牙齿/牙龈边界通常就在它附近，但会有
零点几毫米到一毫米的偏差。

步骤：
1. 初始线：分割结果中该牙的边界（牙颈线），按弧长重采样成 N 个站点；
2. 棱强度：每个顶点取相邻凸边二面角的最大值（弧度）；
3. 候选点：每个站点附近 search_radius mm 内棱强度最高的若干个顶点，外加离站点最近的顶点（允许“原地不动”）；
4. 动态规划：每个站点选一个候选点，总代价 = −棱强度 + 平滑项（相邻选点的距离）+ 沿线滑动项
   （候选点与初始线的高度差不超过 max_height_diff，排除轴壁顶部等其它棱），
   首尾相接形成闭合线；
5. 轻度平滑后输出，同时给出置信度（沿线平均棱强度）。

以后有了颈缘线标注数据，可以换成学习模型（见 docs/architecture.md 的颈缘线调研），接口保持不变。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .cervical import cervical_line


@dataclass
class MarginResult:
    fdi: int
    points: np.ndarray  # (N, 3) 闭合折线，首尾不重复
    confidence: float  # 沿线平均棱强度，归一化到 0~1
    method: str = "edge_snake"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "points": np.round(self.points, 4).tolist(),
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "source": "auto",
            "warnings": self.warnings,
        }


def edge_sharpness(mesh: trimesh.Trimesh, dilate: int = 0) -> np.ndarray:
    """每个顶点相邻凸边的最大二面角（弧度）。凹边（如轴壁与肩台的内角）不算。

    dilate：在一环邻域内取最大值的次数（默认不扩散：实验中扩散会让棱的位置模糊一两格，误差反而变大）。"""
    score = np.zeros(len(mesh.vertices))
    angles = np.where(mesh.face_adjacency_convex, mesh.face_adjacency_angles, 0.0)
    edges = mesh.face_adjacency_edges
    np.maximum.at(score, edges[:, 0], angles)
    np.maximum.at(score, edges[:, 1], angles)
    e = mesh.edges_unique
    for _ in range(dilate):
        grown = score.copy()
        np.maximum.at(grown, e[:, 0], score[e[:, 1]])
        np.maximum.at(grown, e[:, 1], score[e[:, 0]])
        score = grown
    return score


def resample_loop(points: np.ndarray, n: int) -> np.ndarray:
    closed = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, closed[:, k]) for k in range(3)], axis=1)


def detect_margin(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    fdi: int,
    stations: int = 120,
    search_radius: float = 1.5,
    candidates: int = 40,
    smooth_weight: float = 2.0,
    slide_weight: float = 1.0,
    prior_weight: float = 0.0,
    max_height_diff: float = 1.2,
    up: np.ndarray | None = None,
    sharpness: np.ndarray | None = None,
    start_candidates: int = 8,
) -> MarginResult:
    """up：咬合方向（牙冠朝向）。不提供时用该牙表面法向的面积加权平均估计。"""
    initial = resample_loop(cervical_line(mesh, labels, fdi, smooth_iters=3), stations)
    score = edge_sharpness(mesh) if sharpness is None else sharpness
    verts = np.asarray(mesh.vertices)
    if up is None:
        faces = np.all(labels[mesh.faces] == fdi, axis=1)
        up = (mesh.face_normals[faces] * mesh.area_faces[faces, None]).sum(axis=0)
    up = np.asarray(up, dtype=float) / np.linalg.norm(up)

    # 只在初始线附近的顶点里找
    tree = cKDTree(verts)
    band = np.unique(np.concatenate(tree.query_ball_point(initial, search_radius)))
    band_tree = cKDTree(verts[band])

    tangent = np.roll(initial, -1, axis=0) - np.roll(initial, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)

    cand_idx, cand_cost = [], []
    for i, p in enumerate(initial):
        local = band[band_tree.query_ball_point(p, search_radius)]
        # 颈缘线在牙龈附近：高度差太大的（比如轴壁顶部的棱）不考虑
        local = local[np.abs((verts[local] - p) @ up) <= max_height_diff]
        top = local[np.argsort(-score[local])[:candidates]]
        nearest = band[band_tree.query(p)[1]]
        c = np.unique(np.append(top, nearest))
        offset = verts[c] - p
        slide = (offset @ tangent[i]) ** 2
        cand_idx.append(c)
        cand_cost.append(-score[c] + slide_weight * slide + prior_weight * (offset**2).sum(axis=1))

    # 闭合线：在“最明确”的站点（候选里棱强度最高的那个）固定起点，做开放的动态规划，
    # 最后一个站点回到起点的连接也计入平滑代价。起点的候选只取一个时，结果会受起点位置和
    # 初始线走向影响（真实数据上走向反过来，个别点能差 0.5 mm），所以试几个起点候选，取闭合总代价最小的
    start = int(np.argmax([score[c].max() for c in cand_idx]))
    order = [(start + k) % stations for k in range(stations)]
    first = order[0]
    best = (np.inf, None)
    for j0 in np.argsort(cand_cost[first])[:start_candidates]:
        cost = np.full(len(cand_idx[first]), np.inf)
        cost[j0] = cand_cost[first][j0]
        backs = []
        prev = first
        for st in order[1:]:
            a, b = verts[cand_idx[prev]], verts[cand_idx[st]]
            trans = cost[:, None] + smooth_weight * ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
            arg = np.argmin(trans, axis=0)
            cost = trans[arg, np.arange(len(arg))] + cand_cost[st]
            backs.append(arg)
            prev = st
        closing = smooth_weight * ((verts[cand_idx[prev]] - verts[cand_idx[first]][j0]) ** 2).sum(axis=1)
        j = int(np.argmin(cost + closing))
        if cost[j] + closing[j] < best[0]:
            chosen = [j]
            for arg in reversed(backs):
                j = int(arg[j])
                chosen.append(j)
            best = (float(cost[chosen[0]] + closing[chosen[0]]), chosen[::-1])
    chosen = best[1]
    picked = np.array([cand_idx[st][c] for st, c in zip(order, chosen)])
    # 恢复成按初始线顺序排列
    picked = np.roll(picked, start)

    pts = verts[picked].copy()
    for _ in range(2):
        pts = 0.5 * pts + 0.25 * (np.roll(pts, 1, axis=0) + np.roll(pts, -1, axis=0))

    strength = float(np.mean(score[picked]))
    confidence = float(np.clip(strength / (np.pi / 3), 0.0, 1.0))  # 60° 的棱记为满分
    warnings = []
    if confidence < 0.3:
        warnings.append(f"{fdi} 的颈缘线不清晰（置信度 {confidence:.2f}），请人工确认")
    return MarginResult(fdi=fdi, points=pts, confidence=confidence, warnings=warnings)
