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


def _load_scan(scan) -> tuple[trimesh.Trimesh, np.ndarray]:
    if hasattr(scan, "load"):  # datasets.ZipScan：直接从压缩包读
        return scan.load()
    return load_mesh(scan.mesh_path), load_labels(scan.gt_path)[0]


def _measure_scan(args) -> list[tuple[str, tuple[float, float, float], int, int]]:
    """第一遍：每颗牙冠在局部坐标系下的尺寸（近远中宽、颊舌宽、高），不保留网格。"""
    i, scan, min_vertices = args
    try:
        mesh, labels = _load_scan(scan)
    except (OSError, ValueError, KeyError):
        return []
    out = []
    for fdi in sorted(int(v) for v in np.unique(labels) if v != 0):
        if (labels == fdi).sum() < min_vertices or jaw_of(fdi) != scan.jaw:
            continue
        try:
            crown = extract_crown(mesh, labels, scan.jaw, fdi)
        except (ValueError, IndexError):
            continue
        v = crown.vertices
        out.append((template_key(scan.jaw, position(fdi)), (float(np.ptp(v[:, 0])), float(np.ptp(v[:, 1])), float(v[:, 2].max())), i, fdi))
    return out


def build_library(scans, out_dir: str | Path, min_vertices: int = 200, workers: int = 1) -> dict:
    """从带标注的数据集构建模板库：每个牙位取尺寸（近远中宽、颊舌宽、冠高）最接近中位数的那颗真实牙冠（medoid）。

    不做平均：不同牙冠的网格没有对应关系，直接平均会糊掉。分两遍：第一遍只量尺寸（并行，内存只存数字），
    第二遍只重新读入被选中的那几个扫描取出牙冠。
    scans：evaluation.find_scans 的结果，或 datasets.teeth3ds_zip_scans（直接读压缩包，不用解压）。"""
    from concurrent.futures import ProcessPoolExecutor

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(i, scan, min_vertices) for i, scan in enumerate(scans)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            measured = [m for part in pool.map(_measure_scan, jobs, chunksize=4) for m in part]
    else:
        measured = [m for job in jobs for m in _measure_scan(job)]

    candidates: dict[str, list[tuple[np.ndarray, int, int]]] = {}
    for key, dims, i, fdi in measured:
        candidates.setdefault(key, []).append((np.array(dims), i, fdi))

    index = {}
    for key, items in sorted(candidates.items()):
        dims = np.array([d for d, _, _ in items])
        median = np.median(dims, axis=0)
        score = np.linalg.norm((dims - median) / median, axis=1)  # 各尺寸按中位数归一化
        best = int(np.argmin(score))
        _, i, fdi = items[best]
        scan = scans[i]
        mesh, labels = _load_scan(scan)
        crown = extract_crown(mesh, labels, scan.jaw, fdi)
        crown.export(out / f"{key}.stl")
        index[key] = {
            "source": f"{scan.name}:{fdi}",
            "samples": len(items),
            "dims_mm": [round(float(x), 2) for x in dims[best]],
            "median_dims_mm": [round(float(x), 2) for x in median],
            "width_mm": round(float(dims[best][0]), 2),
            "median_width_mm": round(float(median[0]), 2),
        }
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
