"""病例存储。开发阶段用本地目录 + case.json；生产环境换成 PostgreSQL（元数据）+ MinIO/S3（网格文件）。"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(os.environ.get("DENTAL_DATA_DIR", "data/cases"))


def _case_dir(case_id: str) -> Path:
    # case_id 由服务端生成（uuid hex），这里再校验一次防止路径穿越
    if not case_id.isalnum():
        raise KeyError(case_id)
    return DATA_DIR / case_id


def create_case(jaw: str, preferences: dict | None = None) -> dict[str, Any]:
    case_id = uuid.uuid4().hex
    _case_dir(case_id).mkdir(parents=True)
    case = {
        "id": case_id,
        "jaw": jaw,
        "status": "uploaded",  # uploaded -> running -> done | failed
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preferences": preferences or {},
        "files": {},
        "report": None,
        "error": None,
    }
    save_case(case)
    return case


def save_case(case: dict[str, Any]) -> None:
    path = _case_dir(case["id"]) / "case.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(case, ensure_ascii=False))
    tmp.replace(path)


def load_case(case_id: str) -> dict[str, Any]:
    path = _case_dir(case_id) / "case.json"
    if not path.exists():
        raise KeyError(case_id)
    return json.loads(path.read_text())


def list_cases() -> list[dict[str, Any]]:
    if not DATA_DIR.exists():
        return []
    cases = [json.loads(p.read_text()) for p in DATA_DIR.glob("*/case.json")]
    return sorted(cases, key=lambda c: c["created_at"], reverse=True)


def case_path(case_id: str, *parts: str) -> Path:
    base = _case_dir(case_id).resolve()
    path = base.joinpath(*parts).resolve()
    if not path.is_relative_to(base):
        raise KeyError(parts)
    return path
