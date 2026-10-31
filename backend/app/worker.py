"""处理任务。开发阶段由 FastAPI BackgroundTasks 在进程内执行；
生产环境把 process_case 注册为 Celery/RQ 任务，跑在 GPU worker 上，接口不变。"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone

from dental_ai.generation.mirror_template import MirrorTemplateGenerator
from dental_ai.io import load_mesh
from dental_ai.orientation import OrientedSegmenter
from dental_ai.pipeline import run
from dental_ai.preferences import DesignPreferences
from dental_ai.segmentation.base import LabelFileSegmenter, Segmenter
from dental_ai.segmentation.remote import RemoteSegmenter

from . import storage


def build_segmenter(case: dict) -> Segmenter:
    """上传了标签文件就直接用（人工修正结果、Teeth3DS 标注），否则调用分割服务（TGNet，需要 GPU），
    都没有时用 CPU 基线模型（BASELINE_MODEL）。"""
    labels = case["files"].get("labels")
    if labels:
        name = "manual" if case.get("label_edits") else "label_file"
        return LabelFileSegmenter(storage.case_path(case["id"], labels), name=name)
    url = os.environ.get("SEGMENTER_URL")
    if url:
        segmenter: Segmenter = RemoteSegmenter(url, name=os.environ.get("SEGMENTER_NAME", "tgnet"))
        # TGNet 要求标准朝向；输入已经是标准朝向时可以设 SEGMENTER_ORIENT=0 关闭
        if os.environ.get("SEGMENTER_ORIENT", "1") != "0":
            segmenter = OrientedSegmenter(segmenter)
        return segmenter
    deep = os.environ.get("DEEP_MODEL")
    if deep and os.path.exists(deep):
        # 自己训练的深度分割模型（dental_ai/segmentation/deep.py），有 GPU 时用 GPU，否则 CPU
        from dental_ai.segmentation.deep import DeepSegmenter

        return DeepSegmenter(deep)
    baseline = os.environ.get("BASELINE_MODEL")
    if baseline and os.path.exists(baseline):
        # 没有 GPU 分割服务时，用 CPU 基线（梯度提升树，见 dental_ai/segmentation/baseline.py）
        from dental_ai.segmentation.baseline import BaselineSegmenter

        return BaselineSegmenter(baseline)
    raise RuntimeError("未配置分割模型（环境变量 SEGMENTER_URL、DEEP_MODEL 或 BASELINE_MODEL），请上传标签文件、启动 tgnet 服务或训练模型")


def _margin_overrides(case: dict) -> dict[int, dict]:
    """最新一版人工编辑的颈缘线。"""
    name = case["files"].get("margins")
    if not name:
        return {}
    data = json.loads(storage.case_path(case["id"], name).read_text())
    return {int(k): v for k, v in data.items()}


def process_case(case_id: str) -> None:
    case = storage.load_case(case_id)
    case.update(status="running", error=None)
    storage.save_case(case)
    try:
        files = case["files"]
        antagonist = files.get("antagonist")
        prefs = DesignPreferences.from_dict(case.get("preferences"))
        report = run(
            load_mesh(storage.case_path(case_id, files["mesh"])),
            case["jaw"],
            build_segmenter(case),
            storage.case_path(case_id, "result"),
            # 自定义模板库（python -m dental_ai.templates build 生成）；不设置时用内置参数化模板
            generator=MirrorTemplateGenerator(template_dir=os.environ.get("TEMPLATE_DIR"), preferences=prefs),
            antagonist=load_mesh(storage.case_path(case_id, antagonist)) if antagonist else None,
            preferences=prefs,
            fix_numbering=os.environ.get("NUMBERING_FIX", "1") != "0",
            prepared_teeth=case.get("prepared_teeth", []),
            margin_overrides=_margin_overrides(case),
        )
        case.update(status="done", report=report)
    except Exception as exc:  # 任务失败要记录到病例上，前端才能显示
        traceback.print_exc()
        case.update(status="failed", error=str(exc))
    case["finished_at"] = datetime.now(timezone.utc).isoformat()
    storage.save_case(case)
