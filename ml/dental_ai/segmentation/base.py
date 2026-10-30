"""牙齿实例分割接口。

输入一个颌的口扫网格，输出逐顶点 FDI 标签（0 为牙龈），与 Teeth3DS 标注格式一致。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import trimesh

from ..fdi import Jaw
from ..io import load_labels


class Segmenter(ABC):
    name: str = "base"
    # 是否对输出做自动后处理（去碎片等）。人工标注、真值标注不需要
    postprocess: bool = True

    @abstractmethod
    def predict(self, mesh: trimesh.Trimesh, jaw: Jaw) -> np.ndarray: ...


class LabelFileSegmenter(Segmenter):
    """直接读取已有标签文件（Teeth3DS 标注、人工修正结果）。用于开发调试和评估下游模块。"""

    name = "label_file"
    postprocess = False

    def __init__(self, label_path: str | Path, name: str = "label_file"):
        self.label_path = Path(label_path)
        self.name = name

    def predict(self, mesh: trimesh.Trimesh, jaw: Jaw) -> np.ndarray:
        labels, _ = load_labels(self.label_path)
        if len(labels) != len(mesh.vertices):
            raise ValueError(f"标签数量 {len(labels)} 与顶点数 {len(mesh.vertices)} 不一致")
        return labels
