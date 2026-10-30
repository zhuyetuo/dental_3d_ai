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


def test_without_labels_fails_cleanly(client, tmp_path):
    case_id = _upload(client, tmp_path, with_labels=False).json()["id"]
    client.post(f"/api/cases/{case_id}/run")
    case = client.get(f"/api/cases/{case_id}").json()
    assert case["status"] == "failed" and "分割模型" in case["error"]


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
