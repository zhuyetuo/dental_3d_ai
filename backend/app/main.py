from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from dental_ai.fdi import arch_order
from dental_ai.io import load_labels, save_labels
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
    prepared: str | None = Form(None, description="预备体牙号，逗号分隔，如 46,36"),
) -> dict:
    try:
        prefs = DesignPreferences.from_dict(json.loads(preferences) if preferences else None)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"设计参数无效: {exc}") from None
    try:
        prepared_teeth = sorted({int(t) for t in (prepared or "").replace("，", ",").split(",") if t.strip()})
    except ValueError:
        raise HTTPException(400, f"预备体牙号格式错误: {prepared}") from None
    _check_teeth(jaw, prepared_teeth)
    case = storage.create_case(jaw, prefs.to_dict())
    case["prepared_teeth"] = prepared_teeth
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


def _check_teeth(jaw: str, teeth) -> None:
    invalid = sorted(int(t) for t in set(teeth) - set(arch_order(jaw, include_third_molars=True)))
    if invalid:
        raise HTTPException(400, f"{jaw} 不允许的牙号: {invalid}")


class PreparedUpdate(BaseModel):
    teeth: list[int]


@app.put("/api/cases/{case_id}/prepared")
def update_prepared(case_id: str, body: PreparedUpdate) -> dict:
    """设置哪些牙是预备体。之后调用 /run 会为新增的预备体自动检测颈缘线。"""
    case = _get_case(case_id)
    if case["status"] in ("queued", "running"):
        raise HTTPException(409, "任务正在运行，结束后再修改")
    _check_teeth(case["jaw"], body.teeth)
    case["prepared_teeth"] = sorted(set(body.teeth))
    storage.save_case(case)
    return case


class MarginLine(BaseModel):
    points: list[list[float]]  # 网格表面上的闭合折线
    control: list[list[float]] | None = None  # 编辑用的控制点


class MarginsUpdate(BaseModel):
    margins: dict[int, MarginLine]


@app.put("/api/cases/{case_id}/margins")
def update_margins(case_id: str, body: MarginsUpdate) -> dict:
    """保存人工编辑的颈缘线。和标签一样按版本保存，第一次修改前保留自动检测结果（margins_model.json）。"""
    case = _get_case(case_id)
    if case["status"] in ("queued", "running"):
        raise HTTPException(409, "任务正在运行，结束后再保存")
    _check_teeth(case["jaw"], body.margins.keys())
    for fdi, line in body.margins.items():
        if len(line.points) < 4 or any(len(p) != 3 for p in line.points):
            raise HTTPException(400, f"{fdi} 的颈缘线至少要 4 个三维点")
    edits = case.setdefault("margin_edits", [])
    auto = storage.case_path(case_id, "result", "margins.json")
    if not edits and auto.exists():
        shutil.copyfile(auto, storage.case_path(case_id, "margins_model.json"))
        case["files"]["margins_model"] = "margins_model.json"
    # 在上一版的基础上合并：没提交的牙保持不变
    previous = {}
    if case["files"].get("margins"):
        previous = json.loads(storage.case_path(case_id, case["files"]["margins"]).read_text())
    merged = {**previous, **{str(k): {**v.model_dump(), "source": "manual"} for k, v in body.margins.items()}}
    version = len(edits) + 1
    name = f"margins_edit_{version:03d}.json"
    storage.case_path(case_id, name).write_text(json.dumps(merged))
    edits.append({"version": version, "file": name, "created_at": datetime.now(timezone.utc).isoformat(), "teeth": sorted(body.margins)})
    case["files"]["margins"] = name
    case["prepared_teeth"] = sorted(set(case.get("prepared_teeth", [])) | set(body.margins))
    storage.save_case(case)
    return case


class LabelsUpdate(BaseModel):
    labels: list[int]


@app.put("/api/cases/{case_id}/labels")
def update_labels(case_id: str, body: LabelsUpdate) -> dict:
    """保存人工修正后的逐顶点标签。之后调用 /run 会用这份标签重新计算下游步骤。

    每次修改保存为新版本（labels_edit_001.json …）；第一次修改前把模型的原始预测另存为
    labels_model.json，“模型预测 → 人工修正”成对保留，用作训练数据。
    """
    case = _get_case(case_id)
    if case["status"] in ("queued", "running"):
        raise HTTPException(409, "任务正在运行，结束后再保存")
    current = storage.case_path(case_id, "result", "labels.json")
    if not current.exists():
        raise HTTPException(409, "还没有分割结果，请先运行")
    base, _ = load_labels(current)
    new = np.asarray(body.labels, dtype=np.int64)
    if len(new) != len(base):
        raise HTTPException(400, f"标签数量 {len(new)} 与顶点数 {len(base)} 不一致")
    valid = {0, *arch_order(case["jaw"], include_third_molars=True)}
    invalid = sorted(int(v) for v in set(np.unique(new).tolist()) - valid)
    if invalid:
        raise HTTPException(400, f"{case['jaw']} 不允许的牙号: {invalid}")

    edits = case.setdefault("label_edits", [])
    if not edits:
        shutil.copyfile(current, storage.case_path(case_id, "labels_model.json"))
        case["files"]["labels_model"] = "labels_model.json"
        case["model_segmenter"] = (case.get("report") or {}).get("segmenter")
    version = len(edits) + 1
    name = f"labels_edit_{version:03d}.json"
    save_labels(storage.case_path(case_id, name), new, case["jaw"], patient_id=case_id)
    edits.append(
        {
            "version": version,
            "file": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "changed_vertices": int((new != base).sum()),
        }
    )
    case["files"]["labels"] = name
    storage.save_case(case)
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
