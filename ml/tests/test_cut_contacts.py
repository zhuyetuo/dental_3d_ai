import tempfile

import numpy as np
import pytest
import trimesh

from dental_ai.generation import prep_crown as P
from dental_ai.generation.contacts import signed_distance_to_surface
from dental_ai.geometry import estimate_arch_frame, tooth_mesh
from dental_ai.margin.cervical import boundary_loops
from dental_ai.margin.cut import polygon_signed_distance, split_along_zero
from dental_ai.margin.detect import detect_margin
from dental_ai.pipeline import run
from dental_ai.preferences import DesignPreferences
from dental_ai.segmentation.base import Segmenter
from dental_ai.synthetic import synthetic_arch


def test_polygon_signed_distance():
    square = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float)
    pts = np.array([[1, 1], [3, 1], [1, -0.5]])
    d = polygon_signed_distance(pts, square, np.array([True, False, False]))
    assert np.allclose(d, [-1, 1, 0.5])


def test_split_along_zero_keeps_surface_and_splits_cleanly():
    grid = trimesh.creation.box(extents=(4, 4, 1)).subdivide().subdivide()
    f = grid.vertices[:, 0] - 0.3  # 沿 x = 0.3 切开
    cut, values = split_along_zero(grid, f, snap=0.0)
    assert cut.is_watertight  # 切开（拆分三角形）不改变拓扑
    assert abs(cut.area - grid.area) < 1e-9  # 新顶点都在原来的边上
    on_line = cut.vertices[values == 0]
    assert len(on_line) > 0 and np.allclose(on_line[:, 0], 0.3)


def prep_case(fdi, offset):
    mesh, labels = synthetic_arch("lower", prepared=(fdi,), resolution=0.25, band=12, label_offset=offset)
    up = estimate_arch_frame(mesh, labels, "lower").occlusal
    margin = detect_margin(mesh, labels, fdi, up=up).points
    return mesh, labels, up, margin


@pytest.mark.parametrize("fdi", [46, 35, 31, 33])
@pytest.mark.parametrize("offset", [0.4, -0.4])
def test_die_boundary_follows_margin(fdi, offset):
    """精确切割后，预备体表面的边界紧贴颈缘线（不再是线内侧一个三角形）。"""
    mesh, labels, up, margin = prep_case(fdi, offset)
    snapped = trimesh.proximity.closest_point(mesh, margin)[0]
    die = P.extract_die(mesh, snapped, P._frame(snapped, up))
    loop = die.vertices[boundary_loops(die)[0]]
    err = P._distance_to_loop(loop, snapped)
    assert err.mean() < 0.08 and np.percentile(err, 95) < 0.25


@pytest.mark.parametrize("target", [-0.03, 0.1])
def test_prep_crown_contacts(target):
    mesh, labels, up, margin = prep_case(46, 0.4)
    res = P.generate_prep_crown(mesh, labels, "lower", 46, margin, up, preferences=DesignPreferences(contacts_distance=target))
    assert res.mesh.is_watertight and res.info["min_thickness_mm"] >= 0.5 - 1e-3
    for f in (45, 47):
        d = signed_distance_to_surface(res.mesh.vertices, tooth_mesh(mesh, labels, f))[0].min()
        assert abs(d - target) < 0.015, (f, d)


def test_missing_tooth_contacts():
    mesh, labels = synthetic_arch("lower", missing=(36,), band=12)

    class Fixed(Segmenter):
        name = "fixed"
        postprocess = False

        def predict(self, m, jaw):
            return labels

    out = tempfile.mkdtemp()
    report = run(mesh, "lower", Fixed(), out)
    (g,) = report["generated"]
    solid = trimesh.load(f"{out}/{g['file']}")
    assert solid.is_watertight
    for f in (35, 37):
        d = signed_distance_to_surface(solid.vertices, tooth_mesh(mesh, labels, f))[0].min()
        assert abs(d - (-0.03)) < 0.015
        assert g["contacts_mm"][str(f)] == pytest.approx(-0.03, abs=0.015)
