"""网格与标签的读写。

标签格式与 Teeth3DS / 3DTeethSeg'22 保持一致：
    {"id_patient": "...", "jaw": "lower", "labels": [逐顶点 FDI，0 为牙龈], "instances": [...]}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from .fdi import Jaw


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """读取网格，保持顶点顺序不变（逐顶点标签依赖顶点顺序）。

    STL 每个三角形单独存三个顶点，没有共享顶点，读入后需要合并，否则无法求边界和连通性。
    因此 STL 不能和逐顶点标签文件搭配使用，带标签的数据请用 OBJ/PLY（Teeth3DS 用 OBJ）。
    """
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"无法读取为三角网格: {path}")
    if Path(path).suffix.lower() == ".stl":
        mesh.merge_vertices()
    return mesh


def load_labels(path: str | Path) -> tuple[np.ndarray, Jaw | None]:
    data = json.loads(Path(path).read_text())
    return np.asarray(data["labels"], dtype=np.int64), data.get("jaw")


def save_labels(path: str | Path, labels: np.ndarray, jaw: Jaw, patient_id: str = "") -> None:
    labels = np.asarray(labels, dtype=np.int64)
    # instances: 每个连通的牙齿实例一个编号，这里按 FDI 区分即可
    fdi_values = sorted(int(v) for v in np.unique(labels) if v != 0)
    inst_ids = {fdi: i + 1 for i, fdi in enumerate(fdi_values)}
    instances = [inst_ids.get(int(v), 0) for v in labels]
    payload = {"id_patient": patient_id, "jaw": jaw, "labels": labels.tolist(), "instances": instances}
    Path(path).write_text(json.dumps(payload))


def label_colors(labels: np.ndarray) -> np.ndarray:
    """每个 FDI 一个稳定的颜色，牙龈为粉色。返回 (N, 4) uint8。"""
    colors = np.tile(np.array([230, 160, 160, 255], dtype=np.uint8), (len(labels), 1))
    for fdi in np.unique(labels):
        if fdi == 0:
            continue
        rng = np.random.default_rng(int(fdi))
        colors[labels == fdi, :3] = rng.integers(60, 255, size=3)
    return colors


def export_colored_ply(path: str | Path, mesh: trimesh.Trimesh, labels: np.ndarray) -> None:
    colored = mesh.copy()
    colored.visual = trimesh.visual.ColorVisuals(colored, vertex_colors=label_colors(labels))
    colored.export(path)
