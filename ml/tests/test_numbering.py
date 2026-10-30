import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from dental_ai.pipeline import run
from dental_ai.segmentation.base import Segmenter
from dental_ai.segmentation.numbering import correct_numbering, tooth_instances
from dental_ai.synthetic import synthetic_arch


def relabel(labels, mapping):
    out = labels.copy()
    for a, b in mapping.items():
        out[labels == a] = b
    return out


def fix(jaw, missing, mapping, **kw):
    mesh, gt = synthetic_arch(jaw, missing=missing, band=12)
    result = correct_numbering(mesh, relabel(gt, mapping), jaw, canonical_vertices=mesh.vertices, **kw)
    return result, gt


@pytest.mark.parametrize("missing", [(), (34,), (36,), (31, 41)])
def test_keeps_correct_numbering(missing):
    result, gt = fix("lower", missing, {})
    assert not result.applied and (result.labels == gt).all()


@pytest.mark.parametrize(
    "jaw, missing, mapping",
    [
        ("lower", (34,), {35: 34, 36: 35, 37: 36}),  # 缺 34 后整体错位
        ("lower", (44,), {45: 44, 46: 45, 47: 46}),
        ("upper", (24,), {25: 24, 26: 25, 27: 26}),
        ("lower", (), {31: 41, 41: 31}),  # 中切牙左右互换
        ("lower", (), {43: 42}),  # 两颗牙同一个号
    ],
)
def test_fixes_common_errors(jaw, missing, mapping):
    result, gt = fix(jaw, missing, mapping)
    assert result.applied
    assert (result.labels == gt).all()
    assert {(c["from"], c["to"]) for c in result.changes} <= {(b, a) for a, b in mapping.items()} | {(a, a) for a in mapping}


def test_whole_arch_left_right_is_only_warned():
    mapping = {**{30 + p: 40 + p for p in range(1, 8)}, **{40 + p: 30 + p for p in range(1, 8)}}
    result, gt = fix("lower", (), mapping)
    assert not result.applied
    assert any("左右" in w for w in result.warnings)


def test_mirrored_orientation_does_not_break_correct_labels():
    mesh, gt = synthetic_arch("lower", band=12)
    result = correct_numbering(mesh, gt, "lower", canonical_vertices=mesh.vertices * [-1, 1, 1])
    assert not result.applied and (result.labels == gt).all()


def test_robust_to_noisy_boundaries():
    mesh, gt = synthetic_arch("lower", band=12)
    noisy = gt.copy()
    # 每颗牙边界外溢 0.6 mm
    for f in np.unique(gt):
        if f:
            d, _ = cKDTree(mesh.vertices[gt == f]).query(mesh.vertices)
            noisy[(gt == 0) & (d < 0.6)] = f
    result = correct_numbering(mesh, noisy, "lower", canonical_vertices=mesh.vertices)
    assert not result.applied


def test_uses_orientation_when_no_canonical_frame():
    mesh, gt = synthetic_arch("lower", missing=(34,), band=12)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.random(random_state=3).as_matrix()
    mesh.apply_transform(pose)
    result = correct_numbering(mesh, relabel(gt, {35: 34, 36: 35, 37: 36}), "lower")
    assert (result.labels == gt).all()


def test_instances_split_duplicate_labels_and_skip_fragments():
    mesh, gt = synthetic_arch("lower", band=12)
    labels = relabel(gt, {43: 42})
    labels[np.nonzero(gt == 0)[0][:5]] = 45  # 5 个顶点的碎片
    groups = tooth_instances(mesh, labels)
    assert len(groups) == 14


def test_pipeline_reports_numbering_fix(tmp_path):
    mesh, gt = synthetic_arch("lower", missing=(34,), band=12)
    shifted = relabel(gt, {35: 34, 36: 35, 37: 36})

    class Model(Segmenter):
        name = "model"

        def predict(self, m, jaw):
            return shifted

    report = run(mesh, "lower", Model(), tmp_path)
    assert report["numbering"]["applied"]
    assert report["teeth"] == sorted(set(np.unique(gt).tolist()) - {0})
    assert [m["fdi"] for m in report["missing"]] == [34]

    report = run(mesh, "lower", Model(), tmp_path / "off", fix_numbering=False)
    assert report["numbering"] is None and 37 in [m["fdi"] for m in report["missing"]]
