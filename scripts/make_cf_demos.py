"""局部牙弓示例（FDI 16 Crown Dataset 真实预备体）：网页流程要整个牙弓，这里离线算好，网页直接展示。

每例：上颌局部（15、17、16 预备体连同周围牙龈）+ 对颌两颗牙 + 技师设计的牙冠（叠加对比用，original_16）+ 结果：
牙冠外形 = 线上的形状模型（看邻牙、颈缘线、预备体表面、对颌），内表面和封闭 = 网页同一套预备体牙冠代码（颈缘线 = 预备体网格的开边界）。
写到 demo/local/cf/<病例号>/，并在 demo/local/demos.json 里登记（precomputed）。

    python scripts/make_cf_demos.py --data ~/zyt/dental_data/crownfusion/CrownFusion --model models/generation/toothgen_prep.pt \
        --rows runs/shape_prep/cf_test/rows.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from dental_ai.eval_crownfusion import load_case, margin_loop, run_case  # noqa: E402
from dental_ai.generation.base import GeneratedTooth  # noqa: E402
from dental_ai.generation.learned import load_model  # noqa: E402
from dental_ai.generation.prep_crown import generate_prep_crown  # noqa: E402
from dental_ai.io import save_labels  # noqa: E402


class Fixed:
    """把已经算好的外表面交给 generate_prep_crown（它只负责内表面、颈缘对接和封闭）。"""

    accepts_prep = True
    accepts_margin = True

    def __init__(self, crown):
        self.crown = crown

    def generate(self, ctx, target, margin=None, prep=None):
        return GeneratedTooth(fdi=16, crown=self.crown.copy(), method="learned", info={"source": "形状模型（预备体表面 + 颈缘线 + 对颌）"})


def build(model, d: Path, out: Path, device: str, vox=None) -> dict:
    pts, props, prep, crown = load_case(d)
    dent = trimesh.load(d / "dentition_mesh.ply", process=False)
    tree = cKDTree(pts[:, :3])
    up_parts, up_labels, ant_parts = [], [], []
    near_prep = cKDTree(np.asarray(prep.vertices))
    for b in dent.split(only_watertight=False):
        pr = props[tree.query(b.vertices)[1]]
        side, jaw_k = np.bincount(pr[:, 0] + 1).argmax() - 1, np.bincount(pr[:, 1]).argmax()
        if jaw_k == 1:
            ant_parts.append(b)
            continue
        if np.bincount(pr[:, 2]).argmax() == 1:  # 预备体连同周围牙龈：预备体网格上的顶点 = 16，其余 = 牙龈
            lab = np.full(len(b.vertices), 16)  # 数据集去掉了牙龈：这块就是预备体（原来按到预备体顶点的距离标，标成了散点）
        else:
            lab = np.full(len(b.vertices), 15 if side == 1 else 17)
        up_parts.append(b)
        up_labels.append(lab)
    mesh = trimesh.util.concatenate(up_parts)
    labels = np.concatenate(up_labels).astype(np.int64)
    antagonist = trimesh.util.concatenate(ant_parts)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result").mkdir(exist_ok=True)
    mesh.export(out / "mesh.obj")
    save_labels(out / "labels.json", labels, "upper", patient_id=f"fdi16-{d.name}")
    antagonist.export(out / "antagonist.obj")
    crown.export(out / "original_16.stl")

    row, outer = run_case(model, d, 1.0, True, True, device)
    src = "形状模型（预备体表面 + 颈缘线 + 对颌）"
    if vox is not None:  # 体素模型：形状模型的网格贴到体素模型的点上（和线上流程 16 / 26 号一样）
        import torch

        from dental_ai.eval_crownfusion import surface_points
        from dental_ai.generation import voxel_crown as vc

        pp = surface_points(prep, vc.N_PREP, int(d.name) + 1)
        c = vc.crop_center(pp)
        bt = {"prep": torch.tensor(pp - c, dtype=torch.float32)[None],
              "ctx": torch.tensor(np.concatenate([pts[:, :3] - c, (props[:, 1:2] + 1)], 1), dtype=torch.float32)[None]}
        vp, vn = vc.predict_points(vox, vc.make_inputs(bt, "cuda"))[0]
        vp, vn = vp.float().cpu().numpy() + c, vn.float().cpu().numpy()
        v, f = vc.fit_mesh(np.asarray(outer.vertices), np.asarray(outer.faces), [], vp, 150, 0.5)
        v, f = vc.poisson_project(v, f, vp, vn)  # 和线上一样：投到泊松重建的光滑表面上
        outer = trimesh.Trimesh(v, f, process=False)
        if ((outer.triangles_center - v.mean(axis=0)) * outer.face_normals).sum() < 0:
            outer.invert()
        src = "体素模型（预备体 / 邻牙 / 对颌体素 + 逐点精修，泊松重建出光滑表面，贴合形状模型网格）"
    margin = margin_loop(prep)
    warnings = []
    try:
        pc = generate_prep_crown(mesh, labels, "upper", 16, margin, np.array([0.0, 1.0, 0.0]), antagonist=antagonist,
                                 outer_generator=Fixed(outer))
        solid, info, warnings = pc.mesh, pc.info, pc.warnings
        (pc.die if pc.die is not None else prep).export(out / "result" / "prep_16.stl")
    except (LookupError, ValueError) as exc:  # 局部牙弓上缺牙检测等步骤不一定能跑：退回只给外表面
        warnings = [f"完整预备体牙冠没生成（{exc}），只显示形状模型的外表面"]
        solid, info = outer, {}
        prep.export(out / "result" / "prep_16.stl")
    solid.export(out / "result" / "crown_16.stl")
    from dental_ai.io import export_colored_ply

    export_colored_ply(out / "result" / "segmentation.ply", mesh, labels)
    save_labels(out / "result" / "labels.json", labels, "upper", patient_id=f"fdi16-{d.name}")
    (out / "result" / "margins.json").write_text(json.dumps({"16": {"points": margin.tolist(), "source": "dataset"}}))
    (out / "result" / "cervical_lines.json").write_text("{}")
    real = np.asarray(crown.vertices)
    dist = cKDTree(trimesh.sample.sample_surface(solid, 60000, seed=0)[0]).query(real)[0]
    report = {
        "jaw": "upper", "segmenter": "数据集标注", "teeth": [15, 16, 17], "missing": [], "orientation": None, "numbering": None,
        "warnings": warnings + ["局部牙弓（FDI 16 Crown Dataset 真实预备体），结果是离线用线上同一套模型算好的；“原始牙”= 技师设计的牙冠"],
        "generated": [{"fdi": 16, "file": "crown_16.stl", "method": "prep_crown", "watertight": bool(solid.is_watertight), "prep_file": "prep_16.stl",
                       **{k: v for k, v in info.items() if isinstance(v, (int, float, str, bool))}, "source": src}],
        "margins": {"16": {"source": "数据集（预备体网格开边界）", "confidence": None, "method": "dataset"}},
        "comparison": {"16": {"mean_mm": round(float(dist.mean()), 3), "median_mm": round(float(np.median(dist)), 3),
                              "p95_mm": round(float(np.percentile(dist, 95)), 3), "max_mm": round(float(dist.max()), 3),
                              "within_03": round(float((dist <= 0.3).mean()), 3)}},
        "generate_teeth": None,
    }
    (out / "result" / "report.json").write_text(json.dumps(report, ensure_ascii=False))
    return {"case": d.name, "cd": row["cd"], "cd_l2_cf": row["cd_l2_cf"], "rim_to_margin": row["rim_to_margin"], "watertight": bool(solid.is_watertight),
            "warnings": warnings, "vs_technician_mean": report["comparison"]["16"]["mean_mm"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--voxel", type=Path, help="体素模型（generation/voxel_crown.py）：外表面贴到它的点上")
    ap.add_argument("--rows", type=Path, required=True, help="eval_crownfusion 的 rows.json：按距离挑好 / 中等 / 偏难各一例")
    ap.add_argument("--out", type=Path, default=ROOT / "demo" / "local")
    a = ap.parse_args()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(a.model, device)
    vox = None
    if a.voxel:
        from dental_ai.generation import voxel_crown as vc

        vox = vc.load(a.voxel, device)
    rows = sorted(json.loads(a.rows.read_text()), key=lambda r: r["cd"])
    picks = [("效果好", rows[len(rows) // 10]), ("中等", rows[len(rows) // 2]), ("偏难", rows[len(rows) * 9 // 10])]
    entries = []
    for tag, r in picks:
        cid = r["case"]
        dest = a.out / "cf" / cid
        if dest.exists():
            shutil.rmtree(dest)
        info = build(model, a.data / "test" / cid, dest, device, vox)
        print(json.dumps({"tag": tag, **info}, ensure_ascii=False), flush=True)
        entries.append({
            "id": f"cf16-{cid}", "title": f"局部牙弓：真实 16 预备体（{tag}，和技师牙冠差 {info['vs_technician_mean']:.2f} mm）",
            "description": "FDI 16 Crown Dataset 测试集里真实磨好的 16 号预备体，只有 15、17 和对颌两颗牙。牙冠外形由线上的"
                           + ("体素模型（预备体、邻牙、对颌放进 0.16 mm 的体素，逐点精修）" if vox is not None else "形状模型看着邻牙、颈缘线、预备体表面和对颌")
                           + "生成，“原始牙”是技师设计的牙冠，可以叠加对比。网页流程要整个牙弓，这几例是离线用同一个模型算好的",
            "precomputed": f"cf/{cid}", "scan": f"cf/{cid}", "labels": True, "jaw": "upper"})
    p = a.out / "demos.json"
    demos = json.loads(p.read_text()) if p.exists() else []
    demos = [d for d in demos if not d["id"].startswith("cf16-")]
    p.write_text(json.dumps(demos[:3] + entries + demos[3:], ensure_ascii=False, indent=1))
    print("demos", [d["id"] for d in json.loads(p.read_text())])


if __name__ == "__main__":
    main()
