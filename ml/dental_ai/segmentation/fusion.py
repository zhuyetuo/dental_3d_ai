"""两个分割模型合起来用。每个颌两个模型各分一遍，各自做完编号纠错、合并、去小块后数“缺了几颗”。

mode="pick"（默认）：用报缺多的那个的结果（一样多用第一个）。真实缺牙（Bite2Text 190 颌，网页路径）：
  我们的模型 54.9%、微调过的 TeethSplineSeg 60.1%、二选一 62.6%，对照颌误报 20 → 21 颌。
mode="pertooth"：在二选一的基础上逐颗牙补：另一个模型报缺、选中的模型却分出来了的牙，把这块换成另一个模型的分割
  （两个模型的标签逐顶点对应同一个网格）。任一模型报缺就算缺牙的上限是 66.8%。
第二个模型出错（服务没起、超时）时直接用第一个的结果。环境变量 SEGMENTER_FUSION_MODE 选模式。"""

from __future__ import annotations

import os

import numpy as np
import trimesh

from ..fdi import Jaw
from .base import Segmenter


def corrected(mesh: trimesh.Trimesh, labels: np.ndarray, jaw: Jaw) -> tuple[np.ndarray, list[int]]:
    """编号纠错、去小块、合并被切开的牙之后的标签，以及报缺的牙号。"""
    from ..pipeline import correct_numbering, detect_missing, merge_split_teeth, remove_small_islands

    numbering = correct_numbering(mesh, labels, jaw)
    fixed = remove_small_islands(mesh, numbering.labels)
    fixed, _ = merge_split_teeth(mesh, fixed, jaw, points=np.asarray(mesh.vertices))
    return np.asarray(fixed), sorted(int(m.fdi) for m in detect_missing(mesh, fixed, jaw))


def count_missing(mesh: trimesh.Trimesh, labels: np.ndarray, jaw: Jaw) -> list[int]:
    return corrected(mesh, labels, jaw)[1]


class FusionSegmenter(Segmenter):
    def __init__(self, first: Segmenter, second: Segmenter, name: str = "fusion", mode: str | None = None):
        self.first, self.second, self.name = first, second, name
        self.mode = mode or os.environ.get("SEGMENTER_FUSION_MODE", "pick")
        self.choice: dict = {}  # 上一次选了哪个（不叫 last：OrientedSegmenter 的 last 是朝向变换）

    def predict(self, mesh: trimesh.Trimesh, jaw: Jaw) -> np.ndarray:
        a = self.first.predict(mesh, jaw)
        na, nb = getattr(self.first, "name", "first"), getattr(self.second, "name", "second")
        try:
            b = self.second.predict(mesh, jaw)
        except Exception as exc:  # noqa: BLE001
            self.choice = {"chosen": na, "second_error": f"{type(exc).__name__}: {exc}"[:200]}
            return a
        fa, ma = corrected(mesh, a, jaw)
        fb, mb = corrected(mesh, b, jaw)
        use_b = len(mb) > len(ma)
        self.choice = {"chosen": nb if use_b else na, "missing": {na: ma, nb: mb}}
        if self.mode != "pertooth":
            return b if use_b else a
        base, other = (fb.copy(), fa) if use_b else (fa.copy(), fb)
        m_base, m_other = (mb, ma) if use_b else (ma, mb)
        replaced = []
        for f in sorted(set(m_other) - set(m_base)):
            region = base == f
            if region.any():  # 选中的模型把这颗分出来了，另一个模型说它缺：这块用另一个模型的标签
                base[region] = other[region]
                replaced.append(f)
        self.choice["replaced"] = replaced
        return base
