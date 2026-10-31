import io
import json

import numpy as np
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


def _prep_case(client, tmp_path, prepared="46"):
    from dental_ai.synthetic import synthetic_arch as arch

    mesh, labels = arch("lower", prepared=(46,), resolution=0.25, band=12, label_offset=0.5)
    save_labels(tmp_path / "p.json", labels, "lower")
    files = {"mesh": ("lower.ply", io.BytesIO(mesh.export(file_type="ply"))), "labels": ("p.json", (tmp_path / "p.json").open("rb"))}
    case_id = client.post("/api/cases", data={"jaw": "lower", "prepared": prepared}, files=files).json()["id"]
    client.post(f"/api/cases/{case_id}/run")
    return case_id, mesh.metadata["margins"][46]


def test_margin_detection_and_edit(client, tmp_path):
    case_id, (cx, cy, r) = _prep_case(client, tmp_path)
    case = client.get(f"/api/cases/{case_id}").json()
    assert case["status"] == "done", case["error"]
    assert case["prepared_teeth"] == [46]
    assert case["report"]["margins"]["46"]["source"] == "auto"
    auto = json.loads(client.get(f"/api/cases/{case_id}/files/result/margins.json").content)["46"]
    pts = np.array(auto["points"])
    assert np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r).mean() < 0.3
    crowns = [g for g in case["report"]["generated"] if g["method"] == "prep_crown"]
    assert crowns and crowns[0]["file"] == "crown_46.stl" and crowns[0]["watertight"]
    assert client.get(f"/api/cases/{case_id}/files/result/crown_46.stl").status_code == 200

    # 人工编辑：保存一条新线，重新计算后沿用人工版本
    edited = (pts + [0.05, 0, 0]).tolist()
    res = client.put(f"/api/cases/{case_id}/margins", json={"margins": {"46": {"points": edited, "control": edited[::10]}}})
    assert res.status_code == 200, res.text
    assert res.json()["files"]["margins"] == "margins_edit_001.json"
    client.post(f"/api/cases/{case_id}/run")
    case = client.get(f"/api/cases/{case_id}").json()
    assert case["report"]["margins"]["46"]["source"] == "manual"
    rerun = json.loads(client.get(f"/api/cases/{case_id}/files/result/margins.json").content)["46"]
    assert rerun["points"] == edited and rerun["control"] == edited[::10]
    model = json.loads(client.get(f"/api/cases/{case_id}/files/margins_model.json").content)
    assert model["46"]["source"] == "auto"


def test_prepared_and_margin_validation(client, tmp_path):
    case_id, _ = _prep_case(client, tmp_path, prepared="")
    case = client.get(f"/api/cases/{case_id}").json()
    assert case["prepared_teeth"] == [] and case["report"]["margins"] == {}
    # 标记为预备体后重新计算，自动检测
    assert client.put(f"/api/cases/{case_id}/prepared", json={"teeth": [46]}).status_code == 200
    client.post(f"/api/cases/{case_id}/run")
    assert "46" in client.get(f"/api/cases/{case_id}").json()["report"]["margins"]

    assert client.put(f"/api/cases/{case_id}/prepared", json={"teeth": [16]}).status_code == 400  # 下颌不允许上颌牙号
    bad = client.put(f"/api/cases/{case_id}/margins", json={"margins": {"46": {"points": [[0, 0, 0]] * 3}}})
    assert bad.status_code == 400
    res = client.post("/api/cases", data={"jaw": "lower", "prepared": "4x"}, files={"mesh": ("a.ply", b"x")})
    assert res.status_code == 400


def test_adjust_generated_tooth_and_export(client, tmp_path):
    import trimesh

    case_id = _upload(client, tmp_path).json()["id"]
    client.post(f"/api/cases/{case_id}/run")
    before = trimesh.load(io.BytesIO(client.get(f"/api/cases/{case_id}/files/result/tooth_36.stl").content), file_type="stl")
    shift = np.eye(4)
    shift[:3, 3] = [1.0, -0.5, 0.25]
    res = client.put(f"/api/cases/{case_id}/generated/36/transform", json={"matrix": shift.ravel().tolist()})
    assert res.status_code == 200 and "36" in res.json()["adjustments"]
    after = trimesh.load(io.BytesIO(client.get(f"/api/cases/{case_id}/files/result/tooth_36.stl").content), file_type="stl")
    assert np.allclose(after.vertices.mean(axis=0) - before.vertices.mean(axis=0), [1.0, -0.5, 0.25], atol=1e-4)
    # 每次都从原始结果变换，不会累加
    client.put(f"/api/cases/{case_id}/generated/36/transform", json={"matrix": shift.ravel().tolist()})
    again = trimesh.load(io.BytesIO(client.get(f"/api/cases/{case_id}/files/result/tooth_36.stl").content), file_type="stl")
    assert np.allclose(again.vertices, after.vertices, atol=1e-5)
    assert client.put(f"/api/cases/{case_id}/generated/11/transform", json={"matrix": shift.ravel().tolist()}).status_code == 404
    assert client.put(f"/api/cases/{case_id}/generated/36/transform", json={"matrix": [0] * 16}).status_code == 400

    stl = client.get(f"/api/cases/{case_id}/export.stl")
    assert stl.status_code == 200 and len(stl.content) > 1000

    client.post(f"/api/cases/{case_id}/run")  # 重新计算后调整清空，原始备份也删掉
    assert client.get(f"/api/cases/{case_id}").json()["adjustments"] == {}
    from app import storage

    assert not list(storage.case_path(case_id, "result").glob("*.orig.stl"))
