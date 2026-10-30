"""缺牙检测：对比完整牙列，并用相邻牙之间的间隙判断是否需要修复。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ..fdi import Jaw, arch_order, describe, expected_width

GapKind = Literal["space", "closed", "free_end"]


@dataclass
class MissingTooth:
    fdi: int
    kind: GapKind  # space: 有缺隙可修复；closed: 间隙已关闭（如正畸拔牙）；free_end: 游离端缺失
    mesial: int | None  # 近中侧（靠近中线）相邻的现存牙
    distal: int | None  # 远中侧相邻的现存牙
    gap_mm: float | None  # 两侧邻牙表面之间的最小距离
    expected_mm: float
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


def _surface_gap(a: np.ndarray, b: np.ndarray) -> float:
    d, _ = cKDTree(a).query(b, k=1)
    return float(d.min())


def detect_missing(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    jaw: Jaw,
    include_third_molars: bool = False,
    space_ratio: float = 0.5,
) -> list[MissingTooth]:
    """返回缺失牙列表。

    多颗连续缺失时（如 35、36 都缺），共用同一个缺隙，gap_mm 是整段缺隙宽度，
    判断阈值使用这一段所有缺失牙的期望宽度之和。
    """
    order = arch_order(jaw, include_third_molars)
    present = {int(v) for v in np.unique(labels) if v != 0}
    points = {f: mesh.vertices[labels == f] for f in present}

    # 牙弓上最靠中线的位置在 order 的中间
    mid = len(order) // 2
    results: list[MissingTooth] = []
    i = 0
    while i < len(order):
        if order[i] in present:
            i += 1
            continue
        j = i
        while j < len(order) and order[j] not in present:
            j += 1
        run = order[i:j]  # 一段连续缺失
        left = order[i - 1] if i > 0 else None
        right = order[j] if j < len(order) else None
        need = sum(expected_width(f) for f in run)

        if left is None or right is None:
            kind: GapKind = "free_end"
            gap = None
        else:
            gap = _surface_gap(points[left], points[right])
            kind = "space" if gap >= space_ratio * need else "closed"

        for f in run:
            # 患者右侧（order 前半）中线方向在 order 增大的方向
            toward_mid_is_right = order.index(f) < mid
            mesial, distal = (right, left) if toward_mid_is_right else (left, right)
            results.append(
                MissingTooth(
                    fdi=f,
                    kind=kind,
                    mesial=mesial,
                    distal=distal,
                    gap_mm=gap,
                    expected_mm=expected_width(f),
                    description=describe(f),
                )
            )
        i = j
    return results
