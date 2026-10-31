import pickle
from pathlib import Path

import numpy as np
import pytest

from dental_ai.datasets import face_to_vertex_labels, poseidon3d_fdi, safe_unpickle

DATA = Path(__file__).resolve().parents[2] / "data"


def test_face_to_vertex_majority():
    faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    labels = face_to_vertex_labels(faces, np.array([36, 36, 0]), 5)
    assert labels.tolist() == [36, 36, 36, 36, 0]  # 顶点 3：36 和 0 各一票，牙齿优先


def test_poseidon3d_positions_to_fdi():
    # 位置 1–8 患者右侧（远中 → 近中），9–16 左侧；区域 17 是牙龈
    mapping = [-1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1]
    regions = list(range(14)) + [17]
    assert poseidon3d_fdi(regions, mapping, "lower").tolist() == [47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 0]
    assert poseidon3d_fdi([0, 7], mapping, "upper").tolist() == [17, 21]


def test_safe_unpickle_rejects_objects():
    assert safe_unpickle(pickle.dumps((1, [2.5, -1], {"a": (True, None)}))) == (1, [2.5, -1], {"a": (True, None)})
    with pytest.raises(pickle.UnpicklingError):
        safe_unpickle(pickle.dumps(np.zeros(3)))  # 需要加载 numpy 的类


@pytest.mark.skipif(not (DATA / "poseidon3d" / "test-simple").exists(), reason="先用 dental_ai.datasets 导出 Poseidon3D")
def test_poseidon3d_export_matches_teeth3ds_handedness():
    """导出时镜像过：按右手系、牙冠朝 +Z、磨牙在 +Y 摆正后，下颌患者左侧在 +X、上颌在 -X（与 Teeth3DS 相同）。"""
    from dental_ai.evaluation import label_frame
    from dental_ai.fdi import QUADRANTS
    from dental_ai.io import load_labels, load_mesh

    checked = 0
    for obj in sorted((DATA / "poseidon3d" / "test-simple").glob("*/*.obj"))[:5]:
        mesh = load_mesh(obj)
        labels, jaw = load_labels(obj.with_suffix(".json"))
        frame = label_frame(mesh, labels, jaw)
        right_q, left_q = QUADRANTS[jaw]
        pos = labels % 10
        inc = mesh.vertices[(pos == 1) & (labels > 0)].mean(axis=0)
        mol = mesh.vertices[np.isin(pos, [6, 7]) & (labels > 0)].mean(axis=0)
        assert (mol - inc) @ frame[1] > 0  # 用左右定出的 Y 轴与解剖学的前后一致
        checked += 1
    assert checked
