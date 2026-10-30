"""端到端流程：分割 → 缺牙检测 → 生成缺失牙 → 封底 → 导出。

    python -m dental_ai.pipeline lower.stl --jaw lower --labels lower.json --out out/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from .detection.missing import detect_missing
from .fdi import Jaw
from .generation.base import GenerationContext, ToothGenerator
from .generation.mirror_template import MirrorTemplateGenerator
from .geometry import estimate_arch_frame, gum_mesh
from .io import export_colored_ply, load_mesh, save_labels
from .margin.base_surface import close_with_gum_base
from .margin.cervical import all_cervical_lines
from .preferences import DesignPreferences
from .segmentation.base import LabelFileSegmenter, Segmenter
from .segmentation.postprocess import check_arch_order, remove_small_islands


def run(
    mesh: trimesh.Trimesh,
    jaw: Jaw,
    segmenter: Segmenter,
    out_dir: str | Path,
    generator: ToothGenerator | None = None,
    antagonist: trimesh.Trimesh | None = None,
    preferences: DesignPreferences | None = None,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefs = preferences or DesignPreferences()
    generator = generator or MirrorTemplateGenerator(preferences=prefs)

    labels = segmenter.predict(mesh, jaw)
    labels = remove_small_islands(mesh, labels)
    save_labels(out / "labels.json", labels, jaw)
    export_colored_ply(out / "segmentation.ply", mesh, labels)

    report: dict = {
        "jaw": jaw,
        "segmenter": segmenter.name,
        "preferences": prefs.to_dict(),
        "teeth": sorted(int(v) for v in np.unique(labels) if v != 0),
        "warnings": check_arch_order(mesh, labels, jaw),
        "missing": [],
        "generated": [],
    }
    (out / "cervical_lines.json").write_text(json.dumps(all_cervical_lines(mesh, labels)))

    missing = detect_missing(mesh, labels, jaw)
    report["missing"] = [m.to_dict() for m in missing]

    ctx = GenerationContext(mesh=mesh, labels=labels, jaw=jaw, antagonist=antagonist)
    up = estimate_arch_frame(mesh, labels, jaw).occlusal
    gum = gum_mesh(mesh, labels)
    for target in missing:
        if target.kind != "space":
            continue
        try:
            tooth = generator.generate(ctx, target)
            solid = close_with_gum_base(tooth.crown, gum, up, relief=prefs.gum_relief, taper=prefs.emergence_taper)
        except (LookupError, ValueError) as exc:
            report["warnings"].append(f"{target.fdi} 生成失败: {exc}")
            continue
        name = f"tooth_{target.fdi}.stl"
        solid.export(out / name)
        report["generated"].append(
            {"fdi": target.fdi, "file": name, "method": tooth.method, "watertight": bool(solid.is_watertight), **tooth.info}
        )

    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="口扫网格 → 分割 → 缺牙检测 → 生成缺失牙")
    p.add_argument("mesh")
    p.add_argument("--jaw", choices=["upper", "lower"], required=True)
    p.add_argument("--labels", help="Teeth3DS 格式标签文件；不提供时需要配置分割模型")
    p.add_argument("--antagonist", help="对颌网格，用于咬合约束")
    p.add_argument("--templates", help="牙齿模板库目录（<FDI>.stl）")
    p.add_argument("--preferences", help="设计参数 JSON 文件，字段见 dental_ai.preferences")
    p.add_argument("--out", default="out")
    args = p.parse_args()

    if not args.labels:
        p.error("当前版本需要 --labels；接入 TGNet 后见 dental_ai.segmentation.tgnet")
    prefs = DesignPreferences.from_dict(json.loads(Path(args.preferences).read_text()) if args.preferences else None)
    report = run(
        load_mesh(args.mesh),
        args.jaw,
        LabelFileSegmenter(args.labels),
        args.out,
        generator=MirrorTemplateGenerator(template_dir=args.templates, preferences=prefs),
        antagonist=load_mesh(args.antagonist) if args.antagonist else None,
        preferences=prefs,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
