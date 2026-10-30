"""缺失牙生成器接口。所有生成方法（模板、统计形状模型、扩散模型）都实现同一个接口，便于替换和对比。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import trimesh

from ..detection.missing import MissingTooth
from ..fdi import Jaw


@dataclass
class GenerationContext:
    mesh: trimesh.Trimesh
    labels: np.ndarray
    jaw: Jaw
    antagonist: trimesh.Trimesh | None = None  # 对颌，用于咬合约束
    antagonist_labels: np.ndarray | None = None
    # 不能当供体的牙（如预备体：形态已经被磨改）
    excluded_donors: frozenset[int] = frozenset()


@dataclass
class GeneratedTooth:
    fdi: int
    crown: trimesh.Trimesh  # 开放的牙冠外表面（底面由 margin.base_surface 封闭）
    method: str
    info: dict = field(default_factory=dict)


class ToothGenerator(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, ctx: GenerationContext, target: MissingTooth) -> GeneratedTooth: ...
