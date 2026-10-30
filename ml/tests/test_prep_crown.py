import numpy as np
import pytest
import trimesh

from dental_ai.generation.contacts import signed_distance_to_surface
from dental_ai.generation.prep_crown import _distance_to_loop, _frame, extract_die, generate_prep_crown
from dental_ai.geometry import estimate_arch_frame, tooth_mesh
from dental_ai.margin.detect import detect_margin
from dental_ai.pipeline import run
from dental_ai.preferences import DesignPreferences
from dental_ai.segmentation.base import Segmenter
from dental_ai.synthetic import synthetic_arch


def make(fdi, offset=0.4, **kw):
    mesh, labels = synthetic_arch("lower", prepared=(fdi,), resolution=0.25, band=12, label_offset=offset)
    up = estimate_arch_frame(mesh, labels, "lower").occlusal
    margin = detect_margin(mesh, labels, fdi, up=up).points
    return mesh, labels, up, margin, generate_prep_crown(mesh, labels, "lower", fdi, margin, up, **kw)


@pytest.mark.parametrize("fdi, neighbors", [(46, (45, 47)), (35, (34, 36)), (31, (41, 32))])
@pytest.mark.parametrize("offset", [0.4, -0.4])
def test_crown_is_closed_fits_and_is_thick_enough(fdi, neighbors, offset):
    mesh, labels, up, margin, result = make(fdi, offset)
    crown = result.mesh
    assert crown.is_watertight and crown.volume > 0
    assert result.info["min_thickness_mm"] >= 0.5 - 1e-3
    assert not result.warnings

    # 预备体和周围 3 mm 的牙龈、肩台都不能被牙冠穿入（检测点向牙体内移 0.01 mm）
    near = np.union1d(np.nonzero(_distance_to_loop(mesh.vertices, margin) < 3.0)[0], np.nonzero(labels == fdi)[0])
    probe = mesh.vertices[near] - mesh.vertex_normals[near] * 0.01
    inside = crown.contains(probe)
    if inside.any():
        # 允许误差：切割时离切割线 0.03 mm 以内的顶点归到内侧，再加检测点本身的 0.01 mm 偏移
        depth = trimesh.proximity.closest_point(crown, probe[inside])[1].max() + 0.01
        assert depth < 0.05
    # 邻接：默认 -0.03 mm（轻微过接触），邻牙最多被穿入这么深
    for f in neighbors:
        d = signed_distance_to_surface(crown.vertices, tooth_mesh(mesh, labels, f))[0].min()
        assert abs(d - (-0.03)) < 0.015


def test_intaglio_follows_cement_gap():
    mesh, labels, up, margin, result = make(46, preferences=DesignPreferences(cement_gap=0.08))
    die = extract_die(mesh, trimesh.proximity.closest_point(mesh, margin)[0], _frame(margin, up))
    # 离颈缘线 1.5 mm 以上的预备体点，到牙冠表面的距离应当约等于粘接间隙
    far = _distance_to_loop(die.vertices, margin) > 1.5
    d = trimesh.proximity.closest_point(result.mesh, die.vertices[far])[1]
    assert abs(np.median(d) - 0.08) < 0.02


def test_no_neighbor_raises():
    mesh, labels = synthetic_arch("lower", prepared=(37,), resolution=0.25, band=12)
    up = estimate_arch_frame(mesh, labels, "lower").occlusal
    margin = detect_margin(mesh, labels, 37, up=up).points
    with pytest.raises(ValueError, match="邻牙"):
        generate_prep_crown(mesh, labels, "lower", 37, margin, up)


def test_pipeline_generates_prep_crown(tmp_path):
    mesh, labels = synthetic_arch("lower", prepared=(46,), resolution=0.25, band=12, label_offset=0.4)

    class Fixed(Segmenter):
        name = "fixed"
        postprocess = False

        def predict(self, m, jaw):
            return labels

    report = run(mesh, "lower", Fixed(), tmp_path, prepared_teeth=[46])
    crowns = [g for g in report["generated"] if g["method"] == "prep_crown"]
    assert [c["fdi"] for c in crowns] == [46] and crowns[0]["watertight"]
    assert trimesh.load(tmp_path / "crown_46.stl").is_watertight


def test_prepared_teeth_are_not_used_as_donors(tmp_path):
    """缺 36、46 是预备体：36 不能用磨小的 46 镜像，46 也没有对侧供体。"""
    mesh, labels = synthetic_arch("lower", missing=(36,), prepared=(46,), resolution=0.25, band=12, label_offset=0.4)

    class Fixed(Segmenter):
        name = "fixed"
        postprocess = False

        def predict(self, m, jaw):
            return labels

    report = run(mesh, "lower", Fixed(), tmp_path, prepared_teeth=[46])
    assert report["generated"] == []
    assert any("36" in w and "预备体" in w for w in report["warnings"])
