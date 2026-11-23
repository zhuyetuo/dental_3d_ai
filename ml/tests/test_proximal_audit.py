import numpy as np
import pytest
import trimesh

from dental_ai.proximal_audit import audit_proximal


def _neighbor_plane(x: float, half_width: float) -> trimesh.Trimesh:
    return trimesh.Trimesh(
        vertices=[[x, -half_width, -.65], [x, half_width, -.65],
                  [x, half_width, .65], [x, -half_width, .65]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )


def test_closed_crown_supports_bulk_penetration():
    crown = trimesh.creation.box(extents=[2., 2., 2.])
    result = audit_proximal(crown, _neighbor_plane(.8, .65), np.array([0., 0., 1.]), -2.)
    assert result['review_status'] == 'supported_penetration'
    assert result['interior_depth_mm'] == pytest.approx(.2)
    assert result['interior_area_gt_015_mm2'] > .02


def test_open_neighbor_boundary_does_not_count_as_interior_support():
    crown = trimesh.creation.box(extents=[2., 2., 2.])
    result = audit_proximal(crown, _neighbor_plane(.8, .07), np.array([0., 0., 1.]), -2.)
    assert result['review_status'] == 'open_boundary_crossing'
    assert result['interior_area_gt_015_mm2'] == 0
    assert result['boundary_area_gt_005_mm2'] > .02


def test_disjoint_and_non_watertight_crowns():
    crown = trimesh.creation.box(extents=[2., 2., 2.])
    up = np.array([0., 0., 1.])
    neighbor = _neighbor_plane(1.25, .65)
    assert audit_proximal(crown, neighbor, up, -2.)['review_status'] == 'no_supported_penetration'
    crown.update_faces(np.arange(len(crown.faces) - 1))
    assert audit_proximal(crown, neighbor, up, -2.)['review_status'] == 'unmeasurable_mesh'
