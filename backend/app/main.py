from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from dental_ai.preferences import DesignPreferences

from . import storage
from .worker import process_case

app = FastAPI(title="Dental 3D AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MESH_SUFFIXES = {".stl", ".ply", ".obj"}


def _get_case(case_id: str) -> dict:
    try:
        return storage.load_case(case_id)
    except KeyError:
        raise HTTPException(404, "病例不存在") from None


async def _save_upload(case_id: str, upload: UploadFile, stem: str, allowed: set[str]) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"{upload.filename}: 只支持 {', '.join(sorted(allowed))}")
    name = f"{stem}{suffix}"
    storage.case_path(case_id, name).write_bytes(await upload.read())
    return name


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/cases", status_code=201)
async def create_case(
    jaw: Literal["upper", "lower"] = Form(...),
    mesh: UploadFile = File(...),
    labels: UploadFile | None = File(None),
    antagonist: UploadFile | None = File(None),
    preferences: str | None = Form(None, description="设计参数 JSON，字段见 dental_ai.preferences.DesignPreferences"),
) -> dict:
    try:
        prefs = DesignPreferences.from_dict(json.loads(preferences) if preferences else None)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"设计参数无效: {exc}") from None
    case = storage.create_case(jaw, prefs.to_dict())
    files = case["files"]
    files["mesh"] = await _save_upload(case["id"], mesh, "mesh", MESH_SUFFIXES)
    if labels is not None:
        files["labels"] = await _save_upload(case["id"], labels, "labels", {".json"})
    if antagonist is not None:
        files["antagonist"] = await _save_upload(case["id"], antagonist, "antagonist", MESH_SUFFIXES)
    storage.save_case(case)
    return case


@app.get("/api/cases")
def list_cases() -> list[dict]:
    return storage.list_cases()


@app.get("/api/cases/{case_id}")
def get_case(case_id: str) -> dict:
    return _get_case(case_id)


@app.post("/api/cases/{case_id}/run", status_code=202)
def run_case(case_id: str, background: BackgroundTasks) -> dict:
    case = _get_case(case_id)
    if case["status"] == "running":
        raise HTTPException(409, "任务正在运行")
    case["status"] = "queued"
    storage.save_case(case)
    background.add_task(process_case, case_id)
    return case


@app.get("/api/cases/{case_id}/files/{path:path}")
def get_file(case_id: str, path: str) -> FileResponse:
    _get_case(case_id)
    try:
        file = storage.case_path(case_id, path)
    except KeyError:
        raise HTTPException(404) from None
    if not file.is_file() or file.name == "case.json":
        raise HTTPException(404)
    return FileResponse(file)
