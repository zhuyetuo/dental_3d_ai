"""真实口扫（Teeth3DS 官方仓库里的示例病例）上的回归测试。

数据不进 git，先下载：python scripts/download_datasets.py teeth3ds-sample（约 12 MB）。没下载时跳过。
这些测试守住的都是只在真实数据上出现过的问题：孤立顶点、扫描小洞、牙颈线夹点、起伏的牙颈线、
相邻两颗牙连成一块等。
"""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from dental_ai.detection.missing import detect_missing
from dental_ai.fdi import mirror
from dental_ai.geometry import estimate_arch_frame, tooth_mesh
from dental_ai.io import load_labels, load_mesh
from dental_ai.margin.cervical import boundary_loops, cervical_line
from dental_ai.pipeline import run
from dental_ai.segmentation.base import Segmenter
from dental_ai.segmentation.numbering import correct_numbering
from dental_ai.synthetic import simulate_missing

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "teeth3ds_sample"
OBJ = SAMPLE / "obj" / "0EJBIPTC" / "0EJBIPTC_lower.obj"
GT = SAMPLE / "gt" / "0EJBIPTC" / "0EJBIPTC_lower.json"

pytestmark = pytest.mark.skipif(not OBJ.exists(), reason="先运行 python scripts/download_datasets.py teeth3ds-sample")


@pytest.fixture(scope="module")
def scan():
    return load_mesh(OBJ), load_labels(GT)[0]


class Fixed(Segmenter):
    name = "fixed"
    postprocess = False

    def __init__(self, labels):
        self.labels = labels

    def predict(self, m, jaw):
        return self.labels


def test_obj_keeps_all_labelled_vertices(scan):
    mesh, labels = scan
    assert len(mesh.vertices) == len(labels) == 110804  # 其中 7855 个是没有面引用的孤立顶点


def test_cervical_lines_are_full_loops(scan):
    mesh, labels = scan
    for fdi in (31, 33, 36, 46):
        line = cervical_line(mesh, labels, fdi, smooth_iters=0)
        perimeter = np.linalg.norm(line - np.roll(line, -1, axis=0), axis=1).sum()
        assert perimeter > 25  # 以前在夹点处只找到 1–4 mm 的小环
        assert len(boundary_loops(tooth_mesh(mesh, labels, fdi))) == 1


def test_numbering_keeps_ground_truth_and_handles_merged_pair(scan):
    mesh, labels = scan
    assert not correct_numbering(mesh, labels, "lower").applied
    wrong = labels.copy()
    wrong[labels == 45] = 44  # 与 44 连成一块
    wrong[labels == 46] = 45
    result = correct_numbering(mesh, wrong, "lower")
    # 边界上的孤立顶点和小碎片会有零星出入（约 0.2%）
    assert (result.labels[labels == 46] == 46).mean() > 0.99 and (result.labels[labels == 47] == 47).mean() > 0.99
    # 连成一块的 44 + 45 自动拆开
    assert (result.labels[labels == 44] == 44).mean() > 0.85 and (result.labels[labels == 45] == 45).mean() > 0.85


@pytest.mark.parametrize("teeth", [(46,), (36, 46)])
def test_generated_tooth_matches_real_one(scan, tmp_path, teeth):
    mesh0, gt = scan
    mesh, labels = simulate_missing(mesh0, gt, "lower", teeth)
    assert {m.fdi for m in detect_missing(mesh, labels, "lower")} == set(teeth)
    report = run(mesh, "lower", Fixed(labels), tmp_path)
    up = estimate_arch_frame(mesh0, gt, "lower").occlusal
    generated = {g["fdi"]: g for g in report["generated"]}
    for fdi in teeth:
        g = generated[fdi]
        assert g["source"] == (f"mirror:{mirror(fdi)}" if len(teeth) == 1 else "builtin:lower_6")
        solid = trimesh.load(tmp_path / g["file"])  # 按坐标焊接顶点后仍然封闭
        assert solid.is_watertight and solid.is_winding_consistent and solid.volume > 0
        real = tooth_mesh(mesh0, gt, fdi).vertices
        h = real @ up
        crown = real[h > np.percentile(h, 40)]
        pts = trimesh.sample.sample_surface(solid, 20000, seed=0)[0]
        from scipy.spatial import cKDTree

        d = cKDTree(pts).query(crown)[0]
        # 实测：对侧镜像约 0.3 mm，内置模板约 1 mm
        assert d.mean() < (0.5 if len(teeth) == 1 else 1.3)
        assert abs((solid.vertices @ up).max() - h.max()) < 2.0


def test_display_mesh_fills_scan_holes_without_new_vertices(scan, tmp_path):
    from dental_ai.geometry import fill_small_holes

    mesh, _ = scan
    filled = fill_small_holes(mesh)
    assert len(filled.vertices) == len(mesh.vertices)  # 标签仍然一一对应
    assert len(boundary_loops(mesh)) > 4000 and len(boundary_loops(filled)) < 30
    assert filled.is_winding_consistent


def test_prep_crown_on_simulated_real_preparation(scan):
    """真实牙齿磨成肩台预备体（simulate_preparation）：自动检测颈缘线 → 预备体牙冠。

    以前在真实数据上：组织面有几百个洞、切割边界满是尖刺、去倒凹把窝沟点推出 4–5 mm、
    最小厚度迭代把外表面推飞（体积几百万 mm³）。"""
    from scipy.spatial import cKDTree

    from dental_ai.generation.prep_crown import generate_prep_crown
    from dental_ai.margin.detect import detect_margin
    from dental_ai.synthetic import simulate_preparation

    mesh0, gt = scan
    mesh, truth = simulate_preparation(mesh0, gt, "lower", 46)
    up = estimate_arch_frame(mesh, gt, "lower").occlusal
    margin = detect_margin(mesh, gt, 46, up=up)
    assert cKDTree(truth).query(margin.points)[0].mean() < 0.25  # 实测 0.16 mm
    crown = generate_prep_crown(mesh, gt, "lower", 46, margin.points, up)
    assert crown.mesh.is_watertight and not crown.warnings
    assert crown.info["min_thickness_mm"] >= 0.5 - 1e-3
    assert 100 < crown.info["volume_mm3"] < 600
    assert all(abs(v - (-0.03)) < 0.015 for v in crown.info["contacts_mm"].values())
