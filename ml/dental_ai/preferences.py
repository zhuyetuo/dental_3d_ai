"""修复体设计参数。

参数名称和默认值参照 dentalai.ai 批处理界面中 Crown 的 Design Preferences，
单位均为 mm；距离为负表示与邻牙 / 对颌轻微重叠（加工后靠抛光调整到刚好接触）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Literal

# 修复体类别（与 dentalai.ai 批处理类别对应，目前只实现 crown 中“缺失牙”这一种）
Category = Literal["crown", "bridge", "inlay_onlay", "coping", "nightguard", "custom_tray", "model_builder"]


@dataclass
class DesignPreferences:
    contacts_distance: float = -0.03  # 与近远中邻牙的距离
    occlusion_distance: float = 0.30  # 与对颌牙的最小距离
    cement_gap: float = 0.03  # 预备体冠内表面与预备体之间的粘接间隙（场景 B）
    crown_thickness: float = 0.50  # 最小冠厚度（场景 B）
    gum_relief: float = 0.05  # 缺失牙底面到牙龈的距离；<0 为压入牙龈的卵圆形桥体
    emergence_taper: float = 0.85  # 穿龈轮廓收拢比例，越小底面越窄
    undercuts: Literal["remove", "keep"] = "remove"  # 场景 B：去除预备体倒凹
    emergence_profile: Literal["regular", "flat", "convex"] = "regular"
    contacts_shape: Literal["regular", "point", "broad"] = "regular"
    style: Literal["full_contour", "facial_cutback"] = "full_contour"

    @classmethod
    def from_dict(cls, data: dict | None) -> "DesignPreferences":
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"未知的设计参数: {', '.join(sorted(unknown))}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)
