"""处理任务。开发阶段由 FastAPI BackgroundTasks 在进程内执行；
生产环境把 process_case 注册为 Celery/RQ 任务，跑在 GPU worker 上，接口不变。"""

from __future__ import annotations

import traceback

from dental_ai.generation.mirror_template import MirrorTemplateGenerator
from dental_ai.io import load_mesh
from dental_ai.pipeline import run
from dental_ai.preferences import DesignPreferences
from dental_ai.segmentation.base import LabelFileSegmenter, Segmenter

from . import storage


def build_segmenter(case: dict) -> Segmenter:
    labels = case["files"].get("labels")
    if labels:
        return LabelFileSegmenter(storage.case_path(case["id"], labels))
    # TODO: 接入 TGNet / DilatedToothSegNet 后在这里返回模型分割器
    raise RuntimeError("尚未配置分割模型，请随网格一起上传 Teeth3DS 格式的标签文件")


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
            generator=MirrorTemplateGenerator(preferences=prefs),
            antagonist=load_mesh(storage.case_path(case_id, antagonist)) if antagonist else None,
            preferences=prefs,
        )
        case.update(status="done", report=report)
    except Exception as exc:  # 任务失败要记录到病例上，前端才能显示
        traceback.print_exc()
        case.update(status="failed", error=str(exc))
    storage.save_case(case)
