"""咬合调整：让牙冠和对颌的最小距离等于设计参数里的咬合距离，只改需要改的地方。

以前的做法是把整颗牙沿咬合方向往下平移，直到最高的一点离对颌够远。牙尖和对颌的窝沟并不正对，
一个点太高就要整颗牙下移，评分卡（eval_crown）量到的结果是牙冠比邻牙低 0.7 mm、和真牙的形状误差从 0.30 变成 0.40 mm。
技工在设计软件里的做法是两步，这里照做：

1. 够不着对颌时把牙冠抬上去：牙颈一圈不动，往上逐渐过渡到整体上移，抬到比目标再多重叠 overlap；
   差得太远（超过 max_lift，开𬌗、对颌缺牙）不抬；
2. 磨掉穿过目标距离的部分：离对颌不到目标距离的顶点沿咬合方向往下退到刚好满足，周围一圈顶点跟着退一部分，避免磨出台阶。
   先多抬一点再磨，是为了得到几小片接触区，而不是只有一个点接触。
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ..margin.cervical import boundary_loops
from .contacts import signed_distance_to_surface


# 三角形内部的检查点（重心坐标）：中心、三条边的中点、靠近三个角的点。学习模型的牙冠三角形边长有 0.5 mm 左右，只查中心会漏
_BARY = np.array([[1 / 3, 1 / 3, 1 / 3], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])


def _crop(surface: trimesh.Trimesh, points: np.ndarray, reach: float) -> trimesh.Trimesh | None:
    lo, hi = points.min(axis=0) - reach, points.max(axis=0) + reach
    v = np.asarray(surface.vertices)
    inside = np.all((v >= lo) & (v <= hi), axis=1)
    keep = inside[np.asarray(surface.faces)].all(axis=1)
    if keep.sum() < 10:
        return None
    return surface.submesh([np.flatnonzero(keep)], append=True, repair=False)


def adjust_occlusion(
    crown: trimesh.Trimesh,
    antagonist: trimesh.Trimesh,
    up: np.ndarray,
    target: float,
    movable: np.ndarray | None = None,
    max_lift: float = 1.0,
    lift: bool = True,
    overlap: float = 0.15,
    iterations: int = 20,
    tolerance: float = 0.01,
) -> tuple[dict, list[str]]:
    """原地修改 crown.vertices。up：咬合方向（指向对颌）；movable：可以往下磨的顶点（默认全部；预备体牙冠传入“离颈缘线够远”的顶点，颈缘不能动）。
    lift=False：只磨不抬（邻接调整之后复查用）。
    返回 ({"occlusal_lift_mm", "occlusal_reduction_mm", "occlusal_clearance_mm"}, 警告)。"""
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)
    info = {"occlusal_lift_mm": 0.0, "occlusal_reduction_mm": 0.0, "occlusal_clearance_mm": None}
    loops = boundary_loops(crown)
    if loops:
        to_edge = cKDTree(crown.vertices[np.concatenate(loops)]).query(crown.vertices)[0]
    else:
        to_edge = np.full(len(crown.vertices), np.inf)
    if movable is None:
        # 底边一圈也要能往下磨：生成牙冠的底边在邻面处会升到离牙尖顶不到 1 mm，对颌的牙尖正好压在这里。
        # 往下磨是沿咬合方向移动，不改变底边在牙龈上的轮廓，封底不受影响；抬高时这一圈的权重是 0，仍然不动
        movable = np.ones(len(crown.vertices), dtype=bool)
    idx = np.flatnonzero(movable)
    pos = np.full(len(crown.vertices), -1)
    pos[idx] = np.arange(len(idx))
    near = _crop(antagonist, np.asarray(crown.vertices), 5.0)
    if near is None or len(idx) == 0:
        return info, []

    # 只有离对颌近的点才求精确距离：更远的点既不用磨（距离够），也不决定抬高多少（超过 max_lift 就不抬）。
    # 牙冠的轴面离对颌有好几毫米，都求精确距离的话时间几乎全花在它们身上
    reach = target + max_lift + 0.5

    def distance():
        d, _ = signed_distance_to_surface(crown.vertices[idx], near, cutoff=reach)
        return d

    # 只看顶点不够：生成牙冠的三角形有 0.3 mm 大，对颌的牙尖会从顶点之间戳进来。三角形内部也量，缺多少记到它的三个顶点上
    faces = np.asarray(crown.faces)
    faces = faces[movable[faces].all(axis=1)]

    # 往下磨只会让表面离对颌更远，所以每一轮只复查上次全面检查时离目标不到 1 mm 的顶点和三角形（大约十分之一），
    # 这部分到位以后再全面检查一遍。全部都查的话一颗牙冠要 10 秒以上，时间都花在离对颌好几毫米的轴面上
    active = {"v": None, "f": None}

    def shortfall(full: bool = True):
        """每个可动顶点还要往下退多少，以及查过的表面离对颌的最小距离。full=False 只查 active 里的。"""
        vi = np.arange(len(idx)) if full or active["v"] is None else active["v"]
        fi = np.arange(len(faces)) if full or active["f"] is None else active["f"]
        need = np.zeros(len(idx))
        low = np.inf
        d = np.full(len(vi), np.inf)
        if len(vi):
            d, _ = signed_distance_to_surface(crown.vertices[idx[vi]], near, cutoff=reach)
            d = np.where(np.isfinite(d), d, np.inf)
            need[vi] = np.clip(target - d, 0.0, None)
            low = float(d.min())
        df = np.full(len(fi), np.inf)
        if len(fi):
            tri = crown.vertices[faces[fi]]  # (F, 3, 3)
            inner = np.einsum("kj,fjd->kfd", _BARY, tri).reshape(-1, 3)
            dc, _ = signed_distance_to_surface(inner, near, cutoff=reach)
            df = np.where(np.isfinite(dc), dc, np.inf).reshape(len(_BARY), -1).min(axis=0)
            need_c = np.clip(target - df, 0.0, None)
            for k in range(3):
                np.maximum.at(need, pos[faces[fi][:, k]], need_c)
            low = min(low, float(df.min()))
        if full:
            active["v"] = np.flatnonzero(d < target + 1.0)
            active["f"] = np.flatnonzero(df < target + 1.0)
        return need, low

    d = distance()
    if not np.isfinite(d).any():
        return info, []
    d_min = float(d[np.isfinite(d)].min())
    warnings: list[str] = []

    # 1. 抬高
    if lift and d_min > target + tolerance:
        if d_min - target <= max_lift:
            weight = np.clip((to_edge[idx] - 1.0) / 3.0, 0.0, 1.0)  # 牙颈一圈不动，3 mm 以上整体上移
            lifted = 0.0
            # 离对颌最近的地方不一定在权重为 1 的区域（前磨牙牙冠短，牙尖离底边不到 4 mm），抬一次不够就再抬
            for _ in range(4):
                step = min(d_min - target + overlap, max_lift + overlap - lifted)
                crown.vertices[idx] = crown.vertices[idx] + np.outer(weight * step, up)
                lifted += step
                d = distance()
                d_min = float(d[np.isfinite(d)].min()) if np.isfinite(d).any() else np.inf
                if d_min <= target + tolerance or lifted >= max_lift + overlap - 1e-6:
                    break
            info["occlusal_lift_mm"] = round(lifted, 3)
        else:
            warnings.append(f"离对颌 {d_min:.1f} mm，超过可抬高的范围（{max_lift:.1f} mm），没有建立咬合接触")

    # 2. 磨掉穿过目标距离的部分
    edges = crown.edges_unique
    edges = edges[movable[edges].all(axis=1)]
    a, b = pos[edges[:, 0]], pos[edges[:, 1]]
    start = crown.vertices[idx].copy()
    full, low = True, None
    for _ in range(iterations):
        need, found = shortfall(full)
        if need.max() <= tolerance:
            if full:
                low = found
                break
            full = True  # 复查范围内已经到位：全面检查一遍
            continue
        full = False
        move = need.copy()
        for _ in range(2):  # 周围两圈顶点跟着退一部分
            spread = np.zeros_like(move)
            np.maximum.at(spread, a, move[b])
            np.maximum.at(spread, b, move[a])
            move = np.maximum(move, 0.5 * spread)
        crown.vertices[idx] = crown.vertices[idx] - np.outer(move, up)
    if low is None:  # 轮数用完：以全面检查的结果为准
        _, low = shortfall()
    info["occlusal_reduction_mm"] = round(float(((start - crown.vertices[idx]) @ up).max()), 3)
    info["occlusal_clearance_mm"] = round(low, 3) if np.isfinite(low) else None
    if np.isfinite(low) and low < target - 0.05:
        warnings.append(f"与对颌的距离 {low:.2f} mm，未达到目标 {target:.2f} mm")
    return info, warnings
