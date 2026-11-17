"""两个分割模型二选一：每个颌分别用两个模型分割，各自做完编号纠错、合并、去小块后数“缺了几颗”，用报缺多的那个的结果
（一样多用第一个）。真实缺牙（Bite2Text 190 颌）上：我们的模型 54.9%、微调过的 TeethSplineSeg 58.0%、这样选 61.9%，
对照颌（报告没提缺牙）的误报 20 → 21 颌。第二个模型出错（服务没起、超时）时直接用第一个的结果。"""

from __future__ import annotations

import numpy as np
import trimesh

from ..fdi import Jaw
from .base import Segmenter


def count_missing(mesh: trimesh.Trimesh, labels: np.ndarray, jaw: Jaw) -> list[int]:
    from ..pipeline import correct_numbering, detect_missing, merge_split_teeth, remove_small_islands

    numbering = correct_numbering(mesh, labels, jaw)
    fixed = remove_small_islands(mesh, numbering.labels)
    fixed, _ = merge_split_teeth(mesh, fixed, jaw, points=np.asarray(mesh.vertices))
    return sorted(int(m.fdi) for m in detect_missing(mesh, fixed, jaw))


class FusionSegmenter(Segmenter):
    def __init__(self, first: Segmenter, second: Segmenter, name: str = "fusion"):
        self.first, self.second, self.name = first, second, name
        self.choice: dict = {}  # 上一次选了哪个（不叫 last：OrientedSegmenter 的 last 是朝向变换）

    def predict(self, mesh: trimesh.Trimesh, jaw: Jaw) -> np.ndarray:
        a = self.first.predict(mesh, jaw)
        try:
            b = self.second.predict(mesh, jaw)
        except Exception as exc:  # noqa: BLE001
            self.choice = {"chosen": getattr(self.first, "name", "first"), "second_error": f"{type(exc).__name__}: {exc}"[:200]}
            return a
        ma, mb = count_missing(mesh, a, jaw), count_missing(mesh, b, jaw)
        use_b = len(mb) > len(ma)
        self.choice = {"chosen": getattr(self.second if use_b else self.first, "name", "second" if use_b else "first"),
                     "missing": {getattr(self.first, "name", "first"): ma, getattr(self.second, "name", "second"): mb}}
        return b if use_b else a
