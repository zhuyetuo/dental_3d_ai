import json

import numpy as np
import pytest

from dental_ai import fdi
from dental_ai.detection.missing import detect_missing
from dental_ai.generation.base import GenerationContext
from dental_ai.generation.mirror_template import MirrorTemplateGenerator
from dental_ai.geometry import estimate_arch_frame, gum_mesh
from dental_ai.io import load_labels, save_labels
from dental_ai.margin.base_surface import close_with_gum_base
from dental_ai.margin.cervical import cervical_line
from dental_ai.pipeline import run
from dental_ai.segmentation.base import LabelFileSegmenter
from dental_ai.segmentation.postprocess import check_arch_order, remove_small_islands
from dental_ai.synthetic import synthetic_arch


@pytest.fixture(scope="module")
def lower_missing_36():
    return synthetic_arch("lower", missing=(36,))


def test_fdi_helpers():
    assert fdi.mirror(36) == 46 and fdi.mirror(11) == 21
    assert fdi.arch_order("lower")[:2] == [47, 46]
    assert fdi.arch_order("upper")[-1] == 27
    for jaw in ("upper", "lower"):
        for f in fdi.arch_order(jaw, include_third_molars=True):
            assert fdi.class_to_fdi(fdi.fdi_to_class(f), jaw) == f


def test_arch_frame(lower_missing_36):
    mesh, labels = lower_missing_36
    frame = estimate_arch_frame(mesh, labels, "lower")
    assert frame.occlusal @ np.array([0, 0, 1]) > 0.99
    # 下颌患者左侧象限 3 在合成数据里位于 +X
    assert frame.lateral @ np.array([1, 0, 0]) > 0.99
    assert check_arch_order(mesh, labels, "lower") == []


def test_detects_missing_with_space(lower_missing_36):
    mesh, labels = lower_missing_36
    (m,) = detect_missing(mesh, labels, "lower")
    assert m.fdi == 36 and m.kind == "space"
    assert (m.mesial, m.distal) == (35, 37)
    assert m.gap_mm > 0.5 * m.expected_mm


def test_free_end_and_closed_gap():
    mesh, labels = synthetic_arch("lower", missing=(37,))
    (m,) = detect_missing(mesh, labels, "lower")
    assert m.kind == "free_end" and m.distal is None

    # 把 35 的标签改成 34：34 与 36 紧挨着，35 的位置不再有缺隙
    mesh, labels = synthetic_arch("lower")
    labels = np.where(labels == 34, 0, labels)
    labels = np.where(labels == 35, 34, labels)
    found = {m.fdi: m for m in detect_missing(mesh, labels, "lower")}
    assert found[35].kind == "closed"


def test_cervical_line_is_closed_loop(lower_missing_36):
    mesh, labels = lower_missing_36
    line = cervical_line(mesh, labels, 46)
    assert len(line) > 20
    # 牙颈线应位于牙齿底部附近
    tooth_z = mesh.vertices[labels == 46][:, 2]
    assert line[:, 2].mean() < tooth_z.mean()


def test_generate_watertight_tooth(lower_missing_36):
    mesh, labels = lower_missing_36
    (target,) = detect_missing(mesh, labels, "lower")
    ctx = GenerationContext(mesh=mesh, labels=labels, jaw="lower")
    tooth = MirrorTemplateGenerator().generate(ctx, target)
    assert tooth.info["source"] == "mirror:46"

    up = estimate_arch_frame(mesh, labels, "lower").occlusal
    solid = close_with_gum_base(tooth.crown, gum_mesh(mesh, labels), up, relief=0.05)
    assert solid.is_watertight
    assert solid.volume > 0

    # 生成牙应位于 35 与 37 之间
    c = solid.vertices.mean(axis=0)
    c35 = mesh.vertices[labels == 35].mean(axis=0)
    c37 = mesh.vertices[labels == 37].mean(axis=0)
    assert min(c35[0], c37[0]) < c[0] < max(c35[0], c37[0])


def test_remove_small_islands(lower_missing_36):
    mesh, labels = lower_missing_36
    noisy = labels.copy()
    gum_idx = np.nonzero(labels == 0)[0][:5]
    noisy[gum_idx] = 41
    cleaned = remove_small_islands(mesh, noisy)
    assert (cleaned[gum_idx] == 0).all()
    assert (cleaned == labels).mean() > 0.999


def test_pipeline_end_to_end(tmp_path, lower_missing_36):
    mesh, labels = lower_missing_36
    mesh_path = tmp_path / "lower.ply"
    mesh.export(mesh_path)
    save_labels(tmp_path / "lower.json", labels, "lower")
    loaded, jaw = load_labels(tmp_path / "lower.json")
    assert jaw == "lower" and (loaded == labels).all()

    report = run(mesh, "lower", LabelFileSegmenter(tmp_path / "lower.json"), tmp_path / "out")
    assert [g["fdi"] for g in report["generated"]] == [36]
    assert report["generated"][0]["watertight"]
    for name in ("report.json", "labels.json", "segmentation.ply", "cervical_lines.json", "tooth_36.stl"):
        assert (tmp_path / "out" / name).exists()
    lines = json.loads((tmp_path / "out" / "cervical_lines.json").read_text())
    assert "46" in lines and "36" not in lines


@pytest.mark.parametrize("ext", ["obj", "ply"])
def test_load_mesh_keeps_vertex_order(tmp_path, lower_missing_36, ext):
    from dental_ai.io import load_mesh

    mesh, _ = lower_missing_36
    mesh.export(tmp_path / f"a.{ext}")
    loaded = load_mesh(tmp_path / f"a.{ext}")
    assert np.allclose(loaded.vertices, mesh.vertices)


def test_load_stl_merges_vertices(tmp_path, lower_missing_36):
    from dental_ai.io import load_mesh

    mesh, _ = lower_missing_36
    mesh.export(tmp_path / "a.stl")
    assert len(load_mesh(tmp_path / "a.stl").vertices) == len(mesh.vertices)


def test_contacts_distance_controls_width(lower_missing_36):
    from dental_ai.preferences import DesignPreferences

    mesh, labels = lower_missing_36
    (target,) = detect_missing(mesh, labels, "lower")
    ctx = GenerationContext(mesh=mesh, labels=labels, jaw="lower")
    tight = MirrorTemplateGenerator(preferences=DesignPreferences(contacts_distance=-0.1)).generate(ctx, target)
    loose = MirrorTemplateGenerator(preferences=DesignPreferences(contacts_distance=0.5)).generate(ctx, target)
    assert tight.info["scale"] > loose.info["scale"]
    with pytest.raises(ValueError):
        DesignPreferences.from_dict({"bogus": 1})
