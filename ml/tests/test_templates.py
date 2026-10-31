import json
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import trimesh

from dental_ai.geometry import estimate_arch_frame
from dental_ai.io import save_labels
from dental_ai.margin.cervical import boundary_loops
from dental_ai.pipeline import run
from dental_ai.segmentation.base import Segmenter
from dental_ai.synthetic import synthetic_arch
from dental_ai.templates import TemplateLibrary, build_library, extract_crown
from dental_ai.templates.frame import build_frame
from dental_ai.templates.parametric import DIMENSIONS, parametric_crown


@pytest.mark.parametrize("jaw", ["upper", "lower"])
@pytest.mark.parametrize("pos", range(1, 9))
def test_parametric_crowns(jaw, pos):
    m = parametric_crown(jaw, pos)
    md, bl, _ = DIMENSIONS[jaw][pos]
    assert len(boundary_loops(m)) == 1 and m.is_winding_consistent
    assert m.extents[0] == pytest.approx(md, rel=0.02) and m.extents[1] == pytest.approx(bl, rel=0.02)
    assert (m.face_normals[:, 2] * m.area_faces).sum() > 0  # 法向朝外（朝上）
    assert abs(m.vertices[boundary_loops(m)[0]][:, 2]).max() < 1e-9  # 底边在 z = 0


def test_frame_round_trip_and_handedness():
    mesh, labels = synthetic_arch("lower", band=12)
    arch = estimate_arch_frame(mesh, labels, "lower")
    template = parametric_crown("lower", 6)
    for center in ([24.0, 26.0, 4.0], [-24.0, 26.0, 4.0]):  # 患者左侧、右侧
        frame = build_frame(arch, np.array(center), np.array([1.0, -1.0, 0.0]) * np.sign(center[0]) * -1)
        world = frame.to_world(template)
        # 放到口内后法向仍然朝外：牙冠表面整体朝向咬合方向
        assert (world.face_normals @ arch.occlusal * world.area_faces).sum() > 0
        assert np.allclose(frame.to_local(world).vertices, template.vertices)
        # 颊侧朝外
        out = np.asarray(center) - arch.origin
        assert frame.buccal @ out > 0


def test_extract_crown_is_in_local_frame():
    mesh, labels = synthetic_arch("lower", band=12)
    for fdi in (36, 46):  # 左右两侧提取出来的形状应该一致（镜像对称）
        crown = extract_crown(mesh, labels, "lower", fdi)
        assert crown.vertices[:, 2].max() > 4  # 牙冠朝 +z
        assert abs(crown.vertices[:, 0].mean()) < 1.0 and abs(crown.vertices[:, 1].mean()) < 1.5
    a, b = extract_crown(mesh, labels, "lower", 36), extract_crown(mesh, labels, "lower", 46)
    # 合成牙弓本身不是严格左右对称，只要求尺寸接近
    assert np.allclose(np.sort(a.extents), np.sort(b.extents), atol=0.4)


def _dataset(tmp_path):
    data, gt = tmp_path / "obj", tmp_path / "gt"
    for case, missing in (("C1", ()), ("C2", (36,)), ("C3", (31,))):
        mesh, labels = synthetic_arch("lower", missing=missing, band=12)
        (data / case).mkdir(parents=True)
        (gt / case).mkdir(parents=True)
        mesh.export(data / case / f"{case}_lower.obj")
        save_labels(gt / case / f"{case}_lower.json", labels, "lower")
    return data, gt


def test_build_library_and_prefer_it(tmp_path):
    from dental_ai.evaluation import find_scans

    data, gt = _dataset(tmp_path)
    index = build_library(find_scans(data, gt), tmp_path / "lib", min_vertices=100)  # 合成切牙顶点少
    assert set(index) == {f"lower_{p}" for p in range(1, 8)}
    assert index["lower_6"]["samples"] == 5  # C1 两颗 + C2 一颗 + C3 两颗
    lib = TemplateLibrary(tmp_path / "lib")
    assert lib.get(36).source == "template:lower_6"
    assert lib.get(18).source == "builtin:upper_8"  # 库里没有的牙位退回内置模板

    # CLI
    out = tmp_path / "cli"
    subprocess.run([sys.executable, "-m", "dental_ai.templates", "build", "--data-dir", data, "--gt-dir", gt, "--out", out, "--min-vertices", "100"], check=True, capture_output=True)
    assert json.loads((out / "index.json").read_text()).keys() == index.keys()


class Fixed(Segmenter):
    name = "fixed"
    postprocess = False

    def __init__(self, labels):
        self.labels = labels

    def predict(self, m, jaw):
        return self.labels


def _generate(missing, **kw):
    mesh, labels = synthetic_arch("lower", missing=missing, band=12)
    full, full_labels = synthetic_arch("lower", band=12)
    out = tempfile.mkdtemp()
    report = run(mesh, "lower", Fixed(labels), out, **kw)
    result = {}
    for g in report["generated"]:
        solid = trimesh.load(f"{out}/{g['file']}")
        truth = full.vertices[full_labels == g["fdi"]].mean(axis=0)
        result[g["fdi"]] = (g, solid, float(np.linalg.norm(solid.vertices.mean(axis=0)[:2] - truth[:2])))
    return result, report


def test_templates_when_both_sides_missing():
    result, _ = _generate((36, 46))
    for fdi in (36, 46):
        g, solid, err = result[fdi]
        assert g["source"] == "builtin:lower_6" and solid.is_watertight and err < 1.0
        assert all(abs(v + 0.03) < 0.015 for v in g["contacts_mm"].values())


def test_free_end_and_consecutive_missing():
    result, report = _generate((36, 37))
    assert set(result) == {36, 37} and not report["warnings"]
    for fdi, (g, solid, err) in result.items():
        assert g["placement"].startswith("free_end") and solid.is_watertight and err < 0.5
    assert not result[36][1].contains(result[37][1].vertices).any()  # 不重叠

    # 对侧也缺：按训练数据统计的“末端牙相对近中邻牙”的偏移放置（generation/free_end_stats.py）
    result, _ = _generate((37, 47))
    for fdi, (g, solid, err) in result.items():
        assert g["placement"] == "free_end:neighbor_statistics" and g["source"] == "builtin:lower_7" and err < 2.0

    # 连续缺三颗、两侧有邻牙
    result, _ = _generate((34, 35, 36))
    assert all(err < 0.5 for _, _, err in result.values())
