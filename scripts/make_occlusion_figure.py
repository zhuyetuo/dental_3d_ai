"""docs/images/occlusion_fit.png、occlusion_metrics.png：带对颌的生成模型 + 咬合 / 邻接调整的效果图。

    python scripts/make_occlusion_figure.py --model models/generation/toothgen_open.pt --old-model <没有对颌分支的旧模型> --card runs/crown_quality

occlusion_fit：Bite2Text 评分卡里的 4 颗牙（磨牙、前磨牙各 2 颗，按“旧模型不给对颌时离对颌的距离”取 30% 和 70% 分位的病例，不挑好看的）。
每颗牙从咬合面看，按到对颌的距离上色（蓝：有间隙；白：刚好接触；红：穿进对颌）。
occlusion_metrics：评分卡的几个比例，天然牙 / 以前 / 现在。
"""
import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == "scripts" else Path.home() / "zyt/dental_3d_ai"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ml"))
import make_result_figures as F  # noqa: E402
from dental_ai.crown_quality import _crop  # noqa: E402
from dental_ai.eval_crown import _labels, _load  # noqa: E402
from dental_ai.eval_generation import _Fixed  # noqa: E402
from dental_ai.generation.contacts import signed_distance_to_surface  # noqa: E402
from dental_ai.generation.learned import LearnedGenerator  # noqa: E402
from dental_ai.generation.mirror_template import MirrorTemplateGenerator  # noqa: E402
from dental_ai.geometry import estimate_arch_frame, tooth_mesh  # noqa: E402
from dental_ai.pipeline import run  # noqa: E402
from dental_ai.preferences import DesignPreferences  # noqa: E402
from dental_ai.synthetic import simulate_missing  # noqa: E402

# 到对颌的距离 → 颜色：远处浅蓝灰，越近越蓝，刚好接触是黄色，穿进去是红色
STOPS = np.array([-0.5, -0.1, 0.0, 0.1, 0.3, 0.8])
COLORS = np.array([[170, 20, 30], [240, 90, 60], [255, 225, 90], [70, 120, 230], [150, 185, 240], [215, 225, 240]], float)


def fit_colors(mesh, antagonist):
    d, _ = signed_distance_to_surface(np.asarray(mesh.vertices), antagonist, cutoff=3.0)
    d = np.where(np.isfinite(d), d, 3.0)
    return np.stack([np.interp(d, STOPS, COLORS[:, k]) for k in range(3)], axis=1).astype(int), d


def fig_fit(args, out: Path):
    rows_csv = [r for r in csv.DictReader(open(args.card / "crowns.csv")) if r.get("kind") == "generated_no_antagonist" and r.get("occl_clearance") not in (None, "")]
    picks = []
    for name, pos in (("molar", (6,)), ("premolar", (4, 5))):
        sub = sorted((r for r in rows_csv if int(r["fdi"]) % 10 in pos), key=lambda r: float(r["occl_clearance"]))
        picks += [(name, sub[int(q * (len(sub) - 1))]) for q in (0.3, 0.7)]
    old = LearnedGenerator(str(args.old_model), fallback=MirrorTemplateGenerator())
    new = LearnedGenerator(str(args.model), fallback=MirrorTemplateGenerator())
    seg = str(ROOT / "models/segmentation/deep.pt")
    rows, titles, notes = [], [], []
    for name, r in picks:
        pid, fdi, jaw = r["patient"], int(r["fdi"]), r["jaw"]
        other = "lower" if jaw == "upper" else "upper"
        mesh = _load(str(args.zip), pid, jaw)
        lab = _labels(mesh, pid, jaw, seg, args.card / "labels")
        up = estimate_arch_frame(mesh, lab, jaw).occlusal
        real = tooth_mesh(mesh, lab, fdi)
        ant = _crop(_load(str(args.zip), pid, other), np.asarray(real.vertices), 12.0)
        mesh2, lab2 = simulate_missing(mesh, lab, jaw, (fdi,))
        center = real.vertices.mean(axis=0)
        cam = F.camera(up, center - mesh.vertices[lab > 0].mean(axis=0), 80)
        near = np.linalg.norm(mesh2.vertices - center, axis=1) < 16
        ctx_mesh = trimesh.Trimesh(mesh2.vertices, mesh2.faces[near[mesh2.faces].all(axis=1)], process=False)
        col = np.where((lab2 > 0)[:, None], F.IVORY, F.GUM)
        base = F.sample(ctx_mesh, col, per_mm2=500)
        ims, note = [], {"patient": pid, "fdi": fdi}
        variants = (("old", old, None, None), ("raw", new, ant, DesignPreferences(occlusion_distance=None)),
                    ("final", new, ant, DesignPreferences(occlusion_distance=0.0)))
        for key, gen, antagonist, prefs in variants:
            with tempfile.TemporaryDirectory() as tmp:
                rep = run(mesh2, jaw, _Fixed(lab2), tmp, generator=gen, antagonist=antagonist, preferences=prefs)
                g = next(x for x in rep["generated"] if x["fdi"] == fdi)
                solid = trimesh.load(Path(tmp) / g["file"])
            c, d = fit_colors(solid, ant)
            note[key] = round(float(d.min()), 2)
            ims.append(F.render([base, F.sample(solid, c, per_mm2=900)], cam, center, 8))
        c, d = fit_colors(real, ant)
        note["real"] = round(float(d.min()), 2)
        ims.append(F.render([base, F.sample(real, c, per_mm2=900)], cam, center, 8))
        rows.append(ims)
        titles.append(f"{name} {fdi}\nmin distance (mm):\n" + "  ".join(f"{note[k]:+.2f}" for k in ("old", "raw", "final", "real")))
        notes.append(note)
    F.grid(rows, ["before: no antagonist", "new model\n(sees antagonist, no post-fit)", "new model + occlusal fit", "real tooth"], titles, out, panel=3.0)
    print(json.dumps(notes))


def fig_metrics(args, out: Path):
    S = {k: json.loads((ROOT / "runs" / v / "summary.json").read_text()) for k, v in (("before", args.before), ("now", args.card.name))}
    items = [
        ("no occlusal contact\n(gap > 0.2 mm)", "和对颌没有接触（间隙 > 0.2 mm）"),
        ("penetrates antagonist\n(> 0.1 mm)", "穿入对颌超过 0.1 mm"),
        ("open proximal contact\n(gap > 0.15 mm)", "至少一侧邻接有缝（> 0.15 mm）"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (title, key) in zip(axes, items):
        v = [S["now"]["checks"][key]["natural"], S["before"]["checks"][key]["generated"], S["now"]["checks"][key]["generated_contact"]]
        bars = ax.bar(["natural\nteeth", "before", "now"], [100 * x for x in v], color=["#9aa5b1", "#e07b6a", "#3c7deb"])
        ax.bar_label(bars, fmt="%.0f%%")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 105)
        ax.spines[["top", "right"]].set_visible(False)
    m = "ref_chamfer_l1"
    v = [S["before"]["metrics"][m]["generated"]["median"], S["now"]["metrics"][m]["generated_contact"]["median"]]
    bars = axes[3].bar(["before", "now"], v, color=["#e07b6a", "#3c7deb"])
    axes[3].bar_label(bars, fmt="%.2f mm")
    axes[3].set_title("shape error vs real tooth\n(Chamfer, median)", fontsize=10)
    axes[3].set_ylim(0, max(v) * 1.25)
    axes[3].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, default=Path.home() / "zyt/dental_data/Bite2Text.zip")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--old-model", type=Path, required=True)
    p.add_argument("--card", type=Path, required=True, help="新模型的评分卡输出目录（eval_crown --out）")
    p.add_argument("--before", default="crown_quality_contactfix", help="runs/ 下“以前”的评分卡目录名")
    p.add_argument("--out", type=Path, default=ROOT / "docs/images")
    p.add_argument("--only", choices=["fit", "metrics"])
    args = p.parse_args()
    if args.only != "metrics":
        fig_fit(args, args.out / "occlusion_fit.png")
    if args.only != "fit":
        fig_metrics(args, args.out / "occlusion_metrics.png")


if __name__ == "__main__":
    main()
