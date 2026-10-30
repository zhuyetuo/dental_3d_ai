"""FDI 牙位编号工具。

FDI 两位数编号：十位是象限（1 右上、2 左上、3 左下、4 右下），个位是牙位（1 中切牙 … 8 第三磨牙）。
Teeth3DS / 3DTeethSeg'22 的逐顶点标签直接使用 FDI 编号，0 表示牙龈。
"""

from __future__ import annotations

from typing import Literal

Jaw = Literal["upper", "lower"]

GUM = 0

QUADRANTS: dict[str, tuple[int, int]] = {
    # (患者右侧象限, 患者左侧象限)
    "upper": (1, 2),
    "lower": (4, 3),
}

TOOTH_NAMES = {
    1: "中切牙",
    2: "侧切牙",
    3: "尖牙",
    4: "第一前磨牙",
    5: "第二前磨牙",
    6: "第一磨牙",
    7: "第二磨牙",
    8: "第三磨牙",
}

# 平均近远中径（mm），用于判断缺隙是否足以容纳一颗牙。数据取自常用牙体解剖学教材的平均值。
MESIODISTAL_WIDTH_MM: dict[Jaw, dict[int, float]] = {
    "upper": {1: 8.5, 2: 6.5, 3: 7.5, 4: 7.0, 5: 6.5, 6: 10.0, 7: 9.0, 8: 8.5},
    "lower": {1: 5.0, 2: 5.5, 3: 7.0, 4: 7.0, 5: 7.0, 6: 11.0, 7: 10.5, 8: 10.0},
}


def quadrant(fdi: int) -> int:
    return fdi // 10


def position(fdi: int) -> int:
    return fdi % 10


def jaw_of(fdi: int) -> Jaw:
    return "upper" if quadrant(fdi) in (1, 2) else "lower"


def is_valid(fdi: int) -> bool:
    return quadrant(fdi) in (1, 2, 3, 4) and 1 <= position(fdi) <= 8


def mirror(fdi: int) -> int:
    """对侧同名牙：16 <-> 26，36 <-> 46。"""
    q = {1: 2, 2: 1, 3: 4, 4: 3}[quadrant(fdi)]
    return q * 10 + position(fdi)


def arch_order(jaw: Jaw, include_third_molars: bool = False) -> list[int]:
    """沿牙弓从患者右侧最远端到左侧最远端的顺序，如下颌 47..41, 31..37。"""
    right_q, left_q = QUADRANTS[jaw]
    last = 8 if include_third_molars else 7
    right = [right_q * 10 + p for p in range(last, 0, -1)]
    left = [left_q * 10 + p for p in range(1, last + 1)]
    return right + left


def expected_width(fdi: int) -> float:
    return MESIODISTAL_WIDTH_MM[jaw_of(fdi)][position(fdi)]


def fdi_to_class(fdi: int) -> int:
    """FDI -> 0..16 的类别下标（每个颌单独训练时使用）。0 为牙龈。"""
    if fdi == GUM:
        return 0
    offset = 0 if quadrant(fdi) in (1, 3) else 8
    return offset + position(fdi)


def class_to_fdi(cls: int, jaw: Jaw) -> int:
    if cls == 0:
        return GUM
    first_q, second_q = (1, 2) if jaw == "upper" else (3, 4)
    q = first_q if cls <= 8 else second_q
    return q * 10 + (cls - 1) % 8 + 1


def describe(fdi: int) -> str:
    side = "右" if quadrant(fdi) in (1, 4) else "左"
    jaw = "上" if jaw_of(fdi) == "upper" else "下"
    return f"{fdi}（{side}{jaw}{TOOTH_NAMES[position(fdi)]}）"
