"""docs/images/refine_fit.png、refine_metrics.png：“D 出形状、扩散补细节”的效果图。

    python scripts/make_refine_figure.py --card-d runs/crown_quality_d2 --card-new runs/crown_quality_rf06k

refine_fit：Bite2Text 评分卡里的 4 颗牙（磨牙、前磨牙各 2 颗，按 D 的形状误差取 30% 和 70% 分位，不挑好看的）。
三列：D → D + 扩散补细节 → 真牙。上一行从咬合面看、按到对颌的距离上色；下一行从颊侧看（灰色），看牙冠的高度和外形。
refine_metrics：评分卡上天然牙 / D / D+扩散的几个指标。
"""
import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("GEN_DEVICE", "cuda")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == "scripts" else Path.home() / "zyt/dental_3d_ai"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ml"))
import make_result_figures as F  # noqa: E402
from make_occlusion_figure import fit_colors  # noqa: E402
from dental_ai.crown_quality import _crop  # noqa: E402
from dental_ai.eval_crown import _labels, _load  # noqa: E402
from dental_ai.eval_generation import _Fixed  # noqa: E402
from dental_ai.generation.learned import LearnedGenerator  # noqa: E402
from dental_ai.generation.mirror_template import MirrorTemplateGenerator  # noqa: E402
from dental_ai.generation.radial_diffusion import RadialDiffusionGenerator  # noqa: E402
from dental_ai.geometry import estimate_arch_frame, tooth_mesh  # noqa: E402
from dental_ai.pipeline import run  # noqa: E402
from dental_ai.preferences import DesignPreferences  # noqa: E402
from dental_ai.synthetic import simulate_missing  # noqa: E402


def fig_fit(args, out: Path):
    rows_csv = [r for r in csv.DictReader(open(args.card_d / "crowns.csv"))
                if r.get("kind") == "generated_contact" and r.get("ref_chamfer_l1") not in (None, "") and r.get("antagonist_input") in ("True", "1", "true")]
    picks = []
    for name, pos in (("molar", (6, 7)), ("premolar", (4, 5))):
        sub = sorted((r for r in rows_csv if int(r["fdi"]) % 10 in pos), key=lambda r: float(r["ref_chamfer_l1"]))
        picks += [(name, sub[int(q * (len(sub) - 1))]) for q in (0.3, 0.7)]
    d = LearnedGenerator(str(args.d_model), fallback=MirrorTemplateGenerator())
    new = RadialDiffusionGenerator(str(args.radial), fallback=d, refine=args.refine, keep_sigma=args.keep_sigma)
    seg = str(ROOT / "models/segmentation/deep.pt")
    prefs = DesignPreferences(occlusion_distance=0.0)
    rows, titles, notes = [], [], []
    for name, r in picks:
        pid, fdi, jaw = r["patient"], int(r["fdi"]), r["jaw"]
        other = "lower" if jaw == "upper" else "upper"
        mesh = _load(str(args.zip), pid, jaw)
        lab = _labels(mesh, pid, jaw, seg, args.card_d / "labels")
        frame = estimate_arch_frame(mesh, lab, jaw)
        up = frame.occlusal
        real = tooth_mesh(mesh, lab, fdi)
        ant_full = _load(str(args.zip), pid, other)
        ant = _crop(ant_full, np.asarray(real.vertices), 12.0)
        mesh2, lab2 = simulate_missing(mesh, lab, jaw, (fdi,))
        center = real.vertices.mean(axis=0)
        outward = center - mesh.vertices[lab > 0].mean(axis=0)
        outward = outward - up * (outward @ up)
        outward /= np.linalg.norm(outward)
        cam_top = F.camera(up, outward, 80)
        cam_side = F.camera(up, outward, 12)
        near = np.linalg.norm(mesh2.vertices - center, axis=1) < 16
        ctx_mesh = trimesh.Trimesh(mesh2.vertices, mesh2.faces[near[mesh2.faces].all(axis=1)], process=False)
        col = np.where((lab2 > 0)[:, None], F.IVORY, F.GUM)
        base = F.sample(ctx_mesh, col, per_mm2=500)
        solids, note = [], {"patient": pid, "fdi": fdi}
        for key, gen in (("shape", d), ("shape+diffusion", new)):
            with tempfile.TemporaryDirectory() as tmp:
                rep = run(mesh2, jaw, _Fixed(lab2), tmp, generator=gen, antagonist=ant, preferences=prefs)
                g = next(x for x in rep["generated"] if x["fdi"] == fdi)
                solids.append(trimesh.load(Path(tmp) / g["file"]))
                note[key] = g.get("source")
        solids.append(real)
        top, side = [], []
        for s in solids:
            c, _ = fit_colors(s, ant)
            top.append(F.render([base, F.sample(s, c, per_mm2=900)], cam_top, center, 8))
            side.append(F.render([base, F.sample(s, np.tile(F.BLUE, (len(s.vertices), 1)), per_mm2=900)], cam_side, center, 8))
        rows += [top, side]
        titles += [f"{name} {fdi}\nocclusal view", f"{name} {fdi}\nbuccal view"]
        notes.append(note)
    F.grid(rows, ["shape model", "shape model + diffusion detail", "real tooth"], titles, out, panel=2.8)
    print(json.dumps(notes))


def fig_metrics(args, out: Path):
    S = {k: json.loads(v.joinpath("summary.json").read_text()) for k, v in (("D", args.card_d), ("new", args.card_new))}
    K = "generated_contact"
    nat = S["new"]["metrics"]
    items = [
        ("height vs neighbours (mm)\n0 = like natural", lambda s, k: s["metrics"]["height_vs_neighbors"][k]["median"], True),
        ("occlusal contact area (mm²)", lambda s, k: s["metrics"]["occl_contact_area"][k]["median"], True),
        ("articulating-paper area (mm²)", lambda s, k: s["metrics"]["occl_paper_area"][k]["median"], True),
        ("squeezed into a neighbour\n(> 0.15 mm)", lambda s, k: 100 * s["checks"]["至少一侧挤进邻牙（< -0.15 mm）"][k], True),
        ("shape error vs real tooth\n(Chamfer, mm)", lambda s, k: s["metrics"]["ref_chamfer_l1"][k]["median"], False),
    ]
    fig, axes = plt.subplots(1, len(items), figsize=(16, 3.3))
    for ax, (title, get, with_nat) in zip(axes, items):
        labels = (["natural"] if with_nat else []) + ["shape model", "shape + diffusion"]
        vals = ([get(S["new"], "natural")] if with_nat else []) + [get(S["D"], K), get(S["new"], K)]
        colors = (["#9aa5b1"] if with_nat else []) + ["#e07b6a", "#22c7d6"]
        bars = ax.bar(labels, vals, color=colors)
        ax.bar_label(bars, fmt="%.0f%%" if "squeezed" in title else "%.2f")
        ax.set_title(title, fontsize=10)
        lo = min(0, min(vals))
        ax.set_ylim(lo * 1.3 if lo < 0 else 0, max(vals) * 1.25 if max(vals) > 0 else 0.1)
        ax.axhline(0, color="#555", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, default=Path.home() / "zyt/dental_data/Bite2Text.zip")
    p.add_argument("--d-model", type=Path, default=ROOT / "models/generation/toothgen_open.pt")
    p.add_argument("--radial", type=Path, default=ROOT / "models/generation/radial_diffusion.pt")
    p.add_argument("--refine", type=float, default=0.6)
    p.add_argument("--keep-sigma", type=float, default=6.0)
    p.add_argument("--card-d", type=Path, required=True)
    p.add_argument("--card-new", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "docs/images")
    p.add_argument("--only", choices=["fit", "metrics"])
    args = p.parse_args()
    if args.only != "fit":
        fig_metrics(args, args.out / "refine_metrics.png")
    if args.only != "metrics":
        fig_fit(args, args.out / "refine_fit.png")


if __name__ == "__main__":
    main()
