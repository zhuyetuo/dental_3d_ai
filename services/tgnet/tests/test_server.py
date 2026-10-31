"""不依赖 GPU：用假模型测试服务的 HTTP 接口和顶点映射逻辑。"""

import sys
from pathlib import Path

import numpy as np
import trimesh
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import create_app, expand_to_input_vertices  # noqa: E402


def _stub_loader(calls):
    def load():
        calls.append("load")

        def predict(path, jaw):
            # 模拟 TGNet：先去重，再给每个顶点按 x 坐标打标签（上颌编号 11/21）
            mesh = trimesh.load(path, process=False, force="mesh")
            v = np.unique(np.asarray(mesh.vertices), axis=0, return_index=True)[1]
            uniq = np.asarray(mesh.vertices)[np.sort(v)]
            sem = np.where(uniq[:, 0] > 0, 21, 11)
            if jaw == "lower":
                sem = sem + 20
            return sem

        return predict

    return load


def _obj_with_duplicates() -> tuple[bytes, np.ndarray]:
    box = trimesh.creation.box()
    # STL 风格：每个面单独三个顶点 → 大量重复顶点
    verts = box.vertices[box.faces].reshape(-1, 3)
    faces = np.arange(len(verts)).reshape(-1, 3)
    soup = trimesh.Trimesh(verts, faces, process=False)
    return soup.export(file_type="obj").encode(), verts


def test_segment_maps_back_to_input_vertices():
    calls = []
    client = TestClient(create_app(_stub_loader(calls)))
    body, verts = _obj_with_duplicates()
    res = client.post("/segment?jaw=lower", content=body)
    assert res.status_code == 200, res.text
    labels = np.array(res.json()["labels"])
    assert len(labels) == len(verts)
    assert (labels == np.where(verts[:, 0] > 0, 41, 31)).all()

    client.post("/segment?jaw=upper", content=body)
    assert calls == ["load"]  # 模型只加载一次


def test_bad_requests():
    client = TestClient(create_app(_stub_loader([])))
    assert client.post("/segment?jaw=lower", content=b"").status_code == 400
    assert client.post("/segment?jaw=left", content=b"v 0 0 0").status_code == 422


def test_health_reports_missing_weights():
    def broken():
        raise FileNotFoundError("找不到权重文件 /models/tgnet_fps.h5")

    res = TestClient(create_app(broken)).get("/health").json()
    assert res["ok"] is False and "权重" in res["error"]


def test_expand_identity_and_mismatch():
    v = np.random.rand(10, 3)
    labels = np.arange(10)
    assert (expand_to_input_vertices(v, labels) == labels).all()
    v[5] = v[2]
    out = expand_to_input_vertices(v, np.arange(9))
    assert out[5] == out[2] == 2 and out[9] == 8


def test_isolated_vertices_get_nearest_label():
    # 真实 Teeth3DS 扫描里有没被面引用的孤立顶点；TGNet（trimesh 读入）会丢掉它们
    body, verts = _obj_with_duplicates()
    extra = np.array([[0.6, 0.1, 0.1], [-0.6, -0.2, 0.0]])
    head, tail = body.split(b"\nf ", 1)
    body = head + b"".join(b"\nv %f %f %f" % tuple(p) for p in extra) + b"\nf " + tail
    res = TestClient(create_app(_stub_loader([]))).post("/segment?jaw=lower", content=body)
    assert res.status_code == 200, res.text
    labels = np.array(res.json()["labels"])
    assert len(labels) == len(verts) + 2
    assert labels[-2:].tolist() == [41, 31]
