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


def remove_tooth(mesh, labels, teeth):
    """模拟缺牙：把牙齿压到周围牙龈的高度并标成牙龈。"""
    from scipy.spatial import cKDTree

    mesh, labels = mesh.copy(), labels.copy()
    up = estimate_arch_frame(mesh, labels, "lower").occlusal
    v = mesh.vertices
    e = mesh.edges_unique
    for fdi in teeth:
        t = labels == fdi
        cross = e[t[e[:, 0]] != t[e[:, 1]]].ravel()
        ring = np.unique(cross[(labels[cross] == 0)])
        flat = lambda p: p - np.outer(p @ up, up)
        _, idx = cKDTree(flat(v[ring])).query(flat(v[t]), k=6)
        target = (v[ring] @ up)[idx].mean(axis=1) - 0.3
        v[t] += np.outer(target - v[t] @ up, up)
        labels[t] = 0
    mesh.vertices = v
    return mesh, labels


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


@pytest.mark.parametrize("teeth", [(46,), (36, 46)])
def test_generated_tooth_matches_real_one(scan, tmp_path, teeth):
    mesh0, gt = scan
    mesh, labels = remove_tooth(mesh0, gt, teeth)
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
