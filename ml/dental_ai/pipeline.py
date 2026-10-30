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
from .margin.detect import detect_margin, edge_sharpness
from .orientation import OrientedSegmenter
from .preferences import DesignPreferences
from .segmentation.base import LabelFileSegmenter, Segmenter
from .segmentation.numbering import correct_numbering
from .segmentation.postprocess import check_arch_order, remove_small_islands
from .segmentation.remote import RemoteSegmenter


def run(
    mesh: trimesh.Trimesh,
    jaw: Jaw,
    segmenter: Segmenter,
    out_dir: str | Path,
    generator: ToothGenerator | None = None,
    antagonist: trimesh.Trimesh | None = None,
    preferences: DesignPreferences | None = None,
    fix_numbering: bool = True,
    prepared_teeth: list[int] | None = None,
    margin_overrides: dict[int, dict] | None = None,
) -> dict:
    """prepared_teeth：预备体牙号，为它们检测颈缘线。
    margin_overrides：人工编辑过的颈缘线 {牙号: {"points": [...], "control": [...]}}，直接沿用，不再自动检测。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prefs = preferences or DesignPreferences()
    generator = generator or MirrorTemplateGenerator(preferences=prefs)

    labels = segmenter.predict(mesh, jaw)
    orientation = getattr(segmenter, "last", None)
    numbering = None
    if segmenter.postprocess:
        # 编号纠错要在去碎片之前：去碎片会把“同一牙号的第二块”并到邻居里，吞掉被标重号的那颗牙
        if fix_numbering:
            canonical = orientation.apply(mesh).vertices if orientation else None
            numbering = correct_numbering(mesh, labels, jaw, canonical_vertices=canonical)
            labels = numbering.labels
        labels = remove_small_islands(mesh, labels)
    save_labels(out / "labels.json", labels, jaw)
    export_colored_ply(out / "segmentation.ply", mesh, labels)

    report: dict = {
        "jaw": jaw,
        "segmenter": segmenter.name,
        "preferences": prefs.to_dict(),
        "orientation": {"symmetry_error_mm": round(orientation.symmetry_error, 3)} if orientation else None,
        "numbering": numbering.to_dict() if numbering else None,
        "teeth": sorted(int(v) for v in np.unique(labels) if v != 0),
        "warnings": check_arch_order(mesh, labels, jaw) + (numbering.warnings if numbering else []),
        "missing": [],
        "generated": [],
    }
    (out / "cervical_lines.json").write_text(json.dumps(all_cervical_lines(mesh, labels)))

    missing = detect_missing(mesh, labels, jaw)
    report["missing"] = [m.to_dict() for m in missing]

    ctx = GenerationContext(mesh=mesh, labels=labels, jaw=jaw, antagonist=antagonist)
    up = estimate_arch_frame(mesh, labels, jaw).occlusal
    gum = gum_mesh(mesh, labels)

    margins, report["margins"] = {}, {}
    overrides = margin_overrides or {}
    present = set(report["teeth"])
    sharpness = None
    for fdi in sorted(set(prepared_teeth or [])):
        if fdi in overrides:
            margins[str(fdi)] = {**overrides[fdi], "source": "manual"}
        elif fdi not in present:
            report["warnings"].append(f"预备体 {fdi} 不在分割结果中，无法检测颈缘线")
            continue
        else:
            if sharpness is None:
                sharpness = edge_sharpness(mesh)
            try:
                result = detect_margin(mesh, labels, fdi, up=up, sharpness=sharpness)
            except ValueError as exc:
                report["warnings"].append(f"{fdi} 颈缘线检测失败: {exc}")
                continue
            margins[str(fdi)] = result.to_dict()
            report["warnings"] += result.warnings
        m = margins[str(fdi)]
        report["margins"][str(fdi)] = {"source": m["source"], "confidence": m.get("confidence"), "method": m.get("method")}
    (out / "margins.json").write_text(json.dumps(margins))
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
    p.add_argument("--labels", help="Teeth3DS 格式标签文件（直接使用，不调用分割模型）")
    p.add_argument("--segmenter-url", help="分割服务地址，如 http://localhost:8001（services/tgnet）")
    p.add_argument("--no-orient", action="store_true", help="调用分割服务前不做朝向归一化（输入已经是标准朝向时使用）")
    p.add_argument("--no-fix-numbering", action="store_true", help="不做编号纠错")
    p.add_argument("--prepared", type=int, nargs="*", default=[], help="预备体牙号，为它们检测颈缘线，如 --prepared 46 36")
    p.add_argument("--antagonist", help="对颌网格，用于咬合约束")
    p.add_argument("--templates", help="牙齿模板库目录（<FDI>.stl）")
    p.add_argument("--preferences", help="设计参数 JSON 文件，字段见 dental_ai.preferences")
    p.add_argument("--out", default="out")
    args = p.parse_args()

    if args.labels:
        segmenter: Segmenter = LabelFileSegmenter(args.labels)
    elif args.segmenter_url:
        segmenter = RemoteSegmenter(args.segmenter_url, name="tgnet")
        if not args.no_orient:
            segmenter = OrientedSegmenter(segmenter)
    else:
        p.error("需要 --labels 或 --segmenter-url")
    prefs = DesignPreferences.from_dict(json.loads(Path(args.preferences).read_text()) if args.preferences else None)
    report = run(
        load_mesh(args.mesh),
        args.jaw,
        segmenter,
        args.out,
        generator=MirrorTemplateGenerator(template_dir=args.templates, preferences=prefs),
        antagonist=load_mesh(args.antagonist) if args.antagonist else None,
        preferences=prefs,
        fix_numbering=not args.no_fix_numbering,
        prepared_teeth=args.prepared,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
