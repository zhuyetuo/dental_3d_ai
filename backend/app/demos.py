"""网页演示用的示例病例（仓库 demo/ 目录，见 demo/README.md）。

demos.json 里每个示例是一份配置：用哪个原始扫描、去掉哪些牙、哪些牙磨成预备体、是否带标注。
修改版在这里现场生成，不分发（Teeth3DS 是 CC BY-NC-ND 4.0）。
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import numpy as np
from dental_ai.io import parse_obj, save_labels
from dental_ai.synthetic import simulate_missing, simulate_preparation

DEMO_DIR = Path(os.environ.get("DEMO_DIR", Path(__file__).resolve().parents[2] / "demo"))


def segmenter_available() -> bool:
    if os.environ.get("SEGMENTER_URL"):
        return True
    models = [os.environ.get(k) for k in ("DEEP_MODEL", "BASELINE_MODEL")]
    return any(m and os.path.exists(m) for m in models)


def _scan_dir(demo: dict) -> Path:
    return DEMO_DIR / ("local/scans" if demo.get("local") else "scans")


def list_demos() -> list[dict]:
    """仓库里的示例（demo/demos.json）+ 本地示例（demo/local/demos.json，不提交，见 scripts/make_demos.py --local）。"""
    demos = []
    for path, local in ((DEMO_DIR / "demos.json", False), (DEMO_DIR / "local" / "demos.json", True)):
        if path.exists():
            demos += [dict(d, local=local) for d in json.loads(path.read_text())]
    ok = segmenter_available()
    out = []
    for d in demos:
        reason = None
        if not (_scan_dir(d) / f"{d['scan']}.obj.gz").exists():
            reason = "本地示例数据不存在（运行 scripts/make_demos.py --local）"
        elif d.get("requires") == "segmenter" and not ok:
            reason = "需要分割模型（GPU 服务或 CPU 基线）"
        out.append(dict(d, available=reason is None, unavailable_reason=reason))
    return out


def get_demo(demo_id: str) -> dict:
    demo = next((d for d in list_demos() if d["id"] == demo_id), None)
    if demo is None:
        raise KeyError(demo_id)
    return demo


def materialize(demo: dict, target_dir: Path) -> dict[str, str]:
    """生成示例病例的文件，返回 {"mesh": 文件名, "labels": 文件名（可能没有）}。"""
    stem = demo["scan"]
    with gzip.open(_scan_dir(demo) / f"{stem}.obj.gz", "rt", encoding="utf-8", errors="replace") as f:
        mesh = parse_obj(f)
    with gzip.open(_scan_dir(demo) / f"{stem}.json.gz", "rt") as f:
        labels = np.asarray(json.load(f)["labels"], dtype=np.int64)
    jaw = demo["jaw"]
    if demo.get("missing"):
        mesh, labels = simulate_missing(mesh, labels, jaw, demo["missing"])
    for fdi in demo.get("prepared", []):
        mesh, _ = simulate_preparation(mesh, labels, jaw, fdi)
    files = {"mesh": "mesh.obj"}
    mesh.export(target_dir / "mesh.obj")  # trimesh 写 OBJ 保留全部顶点，标签仍一一对应
    if demo.get("labels", True):
        save_labels(target_dir / "labels.json", labels, jaw, patient_id=stem)
        files["labels"] = "labels.json"
    return files

