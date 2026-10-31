import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dental_ai.evaluation import rotation_angle
from dental_ai.orientation import OrientedSegmenter, normalize_orientation
from dental_ai.segmentation.base import Segmenter
from dental_ai.synthetic import synthetic_arch


def canonical(jaw):
    """Teeth3DS 标准坐标系下的合成牙弓（synthetic_arch 已经按这个约定生成）。"""
    return synthetic_arch(jaw, band=12)


def random_pose(seed):
    t = np.eye(4)
    t[:3, :3] = Rotation.random(random_state=seed).as_matrix()
    t[:3, 3] = np.random.default_rng(seed).normal(0, 30, 3)
    return t


@pytest.mark.parametrize("jaw", ["lower", "upper"])
def test_recovers_canonical_frame(jaw):
    mesh, _ = canonical(jaw)
    for seed in range(5):
        pose = random_pose(seed)
        moved = mesh.copy()
        moved.apply_transform(pose)
        result = normalize_orientation(moved, jaw)
        assert rotation_angle(result.transform[:3, :3] @ pose[:3, :3]) < 5
        # 结果以质心为原点
        assert np.allclose(result.apply(moved).vertices.mean(axis=0), 0, atol=1e-6)


def test_oriented_segmenter_passes_canonical_mesh():
    mesh, labels = canonical("lower")
    pose = random_pose(7)
    moved = mesh.copy()
    moved.apply_transform(pose)

    class Probe(Segmenter):
        name = "probe"

        def predict(self, m, jaw):
            self.seen = m
            return labels

    probe = Probe()
    out = OrientedSegmenter(probe).predict(moved, "lower")
    assert (out == labels).all()
    # 内部分割器收到的网格：左侧象限在 +X，磨牙在 +Y，牙冠朝 +Z
    v = probe.seen.vertices
    assert v[labels // 10 == 3][:, 0].mean() > 0 > v[labels // 10 == 4][:, 0].mean()
    assert v[np.isin(labels, [36, 46])][:, 1].mean() > v[np.isin(labels, [31, 41])][:, 1].mean()
    assert v[labels != 0][:, 2].mean() > v[labels == 0][:, 2].mean()
    assert moved.vertices is not probe.seen.vertices  # 原网格不被修改
