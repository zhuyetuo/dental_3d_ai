import io
import json

import pytest
from fastapi.testclient import TestClient

from dental_ai.io import save_labels
from dental_ai.synthetic import synthetic_arch


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from app import storage

    monkeypatch.setattr(storage, "DATA_DIR", tmp_path / "cases")
    from app.main import app

    return TestClient(app)


def _upload(client, tmp_path, with_labels=True, preferences=None):
    mesh, labels = synthetic_arch("lower", missing=(36,))
    mesh_bytes = mesh.export(file_type="ply")
    save_labels(tmp_path / "l.json", labels, "lower")
    files = {"mesh": ("lower.ply", io.BytesIO(mesh_bytes))}
    if with_labels:
        files["labels"] = ("lower.json", (tmp_path / "l.json").open("rb"))
    data = {"jaw": "lower"}
    if preferences is not None:
        data["preferences"] = preferences
    return client.post("/api/cases", data=data, files=files)


def test_full_flow(client, tmp_path):
    res = _upload(client, tmp_path)
    assert res.status_code == 201
    case_id = res.json()["id"]

    # TestClient 会在请求返回前执行 BackgroundTasks
    assert client.post(f"/api/cases/{case_id}/run").status_code == 202
    case = client.get(f"/api/cases/{case_id}").json()
    assert case["status"] == "done", case["error"]
    assert [g["fdi"] for g in case["report"]["generated"]] == [36]

    stl = client.get(f"/api/cases/{case_id}/files/result/tooth_36.stl")
    assert stl.status_code == 200 and len(stl.content) > 1000
    lines = client.get(f"/api/cases/{case_id}/files/result/cervical_lines.json")
    assert "46" in json.loads(lines.content)
    assert [c["id"] for c in client.get("/api/cases").json()] == [case_id]


def test_without_labels_or_segmenter_fails_cleanly(client, tmp_path, monkeypatch):
    monkeypatch.delenv("SEGMENTER_URL", raising=False)
    case_id = _upload(client, tmp_path, with_labels=False).json()["id"]
    client.post(f"/api/cases/{case_id}/run")
    case = client.get(f"/api/cases/{case_id}").json()
    assert case["status"] == "failed" and "SEGMENTER_URL" in case["error"]


def test_uses_segmentation_service(client, tmp_path, monkeypatch, fake_segmenter_url):
    monkeypatch.setenv("SEGMENTER_URL", fake_segmenter_url)
    case_id = _upload(client, tmp_path, with_labels=False).json()["id"]
    client.post(f"/api/cases/{case_id}/run")
    case = client.get(f"/api/cases/{case_id}").json()
    assert case["status"] == "done", case["error"]
    assert case["report"]["segmenter"] == "tgnet"
    assert [g["fdi"] for g in case["report"]["generated"]] == [36]


def test_rejects_bad_input(client, tmp_path):
    res = client.post("/api/cases", data={"jaw": "lower"}, files={"mesh": ("x.exe", b"123")})
    assert res.status_code == 400
    assert client.get("/api/cases/nope").status_code == 404
    case_id = _upload(client, tmp_path).json()["id"]
    assert client.get(f"/api/cases/{case_id}/files/case.json").status_code == 404
    assert client.get(f"/api/cases/{case_id}/files/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_preferences(client, tmp_path):
    res = _upload(client, tmp_path, preferences=json.dumps({"contacts_distance": 0.1, "gum_relief": -0.2}))
    case_id = res.json()["id"]
    assert res.json()["preferences"]["gum_relief"] == -0.2
    client.post(f"/api/cases/{case_id}/run")
    report = client.get(f"/api/cases/{case_id}").json()["report"]
    assert report["preferences"]["contacts_distance"] == 0.1

    assert _upload(client, tmp_path, preferences='{"bogus": 1}').status_code == 400
    assert _upload(client, tmp_path, preferences="not json").status_code == 400


def _done_case(client, tmp_path):
    case_id = _upload(client, tmp_path).json()["id"]
    client.post(f"/api/cases/{case_id}/run")
    labels = json.loads(client.get(f"/api/cases/{case_id}/files/result/labels.json").content)["labels"]
    return case_id, labels


def test_edit_labels_and_rerun(client, tmp_path):
    case_id, labels = _done_case(client, tmp_path)
    edited = [32 if v == 31 else v for v in labels]  # 把 31 改成 32
    res = client.put(f"/api/cases/{case_id}/labels", json={"labels": edited})
    assert res.status_code == 200, res.text
    case = res.json()
    assert case["files"]["labels"] == "labels_edit_001.json"
    assert case["label_edits"][0]["changed_vertices"] == labels.count(31)
    assert case["model_segmenter"] == "label_file"

    client.post(f"/api/cases/{case_id}/run")
    case = client.get(f"/api/cases/{case_id}").json()
    assert case["status"] == "done", case["error"]
    assert case["report"]["segmenter"] == "manual"
    assert 31 not in case["report"]["teeth"]
    rerun = json.loads(client.get(f"/api/cases/{case_id}/files/result/labels.json").content)["labels"]
    assert rerun == edited  # 人工标注不做自动后处理

    # 第二次修改：版本递增，模型原始预测保持不变
    case = client.put(f"/api/cases/{case_id}/labels", json={"labels": labels}).json()
    assert [e["version"] for e in case["label_edits"]] == [1, 2]
    model = json.loads(client.get(f"/api/cases/{case_id}/files/labels_model.json").content)["labels"]
    assert model == labels


def test_edit_labels_validation(client, tmp_path):
    case_id = _upload(client, tmp_path).json()["id"]
    assert client.put(f"/api/cases/{case_id}/labels", json={"labels": [0]}).status_code == 409  # 还没运行

    case_id, labels = _done_case(client, tmp_path)
    assert client.put(f"/api/cases/{case_id}/labels", json={"labels": labels[:-1]}).status_code == 400
    wrong_jaw = [11 if v else 0 for v in labels]  # 下颌不允许上颌牙号
    res = client.put(f"/api/cases/{case_id}/labels", json={"labels": wrong_jaw})
    assert res.status_code == 400 and "11" in res.json()["detail"]
    assert client.put(f"/api/cases/{case_id}/labels", json={"labels": [99] * len(labels)}).status_code == 400
