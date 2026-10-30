"""标准牙模板库。

目录结构：
    <dir>/index.json              每个模板的来源、样本数、尺寸
    <dir>/<jaw>_<position>.stl     牙齿局部坐标系下的牙冠（如 lower_6.stl，同时用于 36 和 46）

查找顺序：自定义目录 → 内置参数化模板。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from ..fdi import Jaw, jaw_of, position
from ..geometry import estimate_arch_frame, tooth_mesh
from ..io import load_labels, load_mesh
from .frame import tooth_frame
from .parametric import parametric_crown


def template_key(jaw: Jaw, pos: int) -> str:
    return f"{jaw}_{pos}"


@dataclass
class Template:
    mesh: trimesh.Trimesh  # 局部坐标系
    source: str

    @property
    def width(self) -> float:
        return float(np.ptp(self.mesh.vertices[:, 0]))


class TemplateLibrary:
    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory) if directory else None
        self._cache: dict[str, Template] = {}

    def get(self, fdi: int) -> Template:
        jaw, pos = jaw_of(fdi), position(fdi)
        key = template_key(jaw, pos)
        if key not in self._cache:
            path = self.directory / f"{key}.stl" if self.directory else None
            if path and path.exists():
                self._cache[key] = Template(load_mesh(path), f"template:{key}")
            else:
                self._cache[key] = Template(parametric_crown(jaw, pos), f"builtin:{key}")
        t = self._cache[key]
        return Template(t.mesh.copy(), t.source)


def extract_crown(mesh: trimesh.Trimesh, labels: np.ndarray, jaw: Jaw, fdi: int) -> trimesh.Trimesh:
    """从已分割的病例中取出一颗牙冠，变换到局部坐标系。"""
    arch = estimate_arch_frame(mesh, labels, jaw)
    frame = tooth_frame(mesh, labels, jaw, fdi, arch)
    return frame.to_local(tooth_mesh(mesh, labels, fdi))


def build_library(scans, out_dir: str | Path, min_vertices: int = 200) -> dict:
    """从 Teeth3DS 格式的数据集构建模板库：每个牙位取近远中宽度最接近中位数的那颗牙冠。

    scans：evaluation.find_scans 的结果（每项有 mesh_path、gt_path、jaw）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, list[tuple[float, trimesh.Trimesh, str]]] = {}
    for scan in scans:
        mesh = load_mesh(scan.mesh_path)
        labels, _ = load_labels(scan.gt_path)
        for fdi in sorted(int(v) for v in np.unique(labels) if v != 0):
            if (labels == fdi).sum() < min_vertices or jaw_of(fdi) != scan.jaw:
                continue
            try:
                crown = extract_crown(mesh, labels, scan.jaw, fdi)
            except (ValueError, IndexError):
                continue
            key = template_key(scan.jaw, position(fdi))
            candidates.setdefault(key, []).append((float(np.ptp(crown.vertices[:, 0])), crown, f"{scan.name}:{fdi}"))

    index = {}
    for key, items in sorted(candidates.items()):
        widths = np.array([w for w, _, _ in items])
        median = float(np.median(widths))
        w, crown, src = min(items, key=lambda it: abs(it[0] - median))
        crown.export(out / f"{key}.stl")
        index[key] = {"source": src, "samples": len(items), "width_mm": round(w, 2), "median_width_mm": round(median, 2)}
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
    return index


def export_builtin(out_dir: str | Path) -> list[str]:
    """把内置参数化模板导出成 STL，便于查看。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for jaw in ("upper", "lower"):
        for pos in range(1, 9):
            name = f"{template_key(jaw, pos)}.stl"
            parametric_crown(jaw, pos).export(out / name)
            names.append(name)
    return names
