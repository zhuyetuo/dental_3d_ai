"""生成 docs/results.md 用的效果图（docs/images/*.png）。在跑完训练和评估的 GPU 服务器上运行：

    python scripts/make_result_figures.py [--old-seg <旧的分割模型，用于“修复前”的对比>]

需要 data/fair_test、data/prep_margin、runs/gen_compare、runs/seg_*、models/ 下的三个模型。
图里的文字用英文（服务器上一般没有中文字体），中文说明写在 docs/results.md 里。
渲染不依赖显卡和 OpenGL：在网格表面密集采样，按深度排序后直接画点（离相机近的后画）。
病例不是挑好看的：生成牙的例子取各场景里 Chamfer 距离最接近中位数和第 90 百分位的病例。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from dental_ai.geometry import estimate_arch_frame, tooth_mesh  # noqa: E402
from dental_ai.io import load_labels, load_mesh  # noqa: E402
from dental_ai.synthetic import simulate_missing  # noqa: E402

GUM = np.array([226, 160, 150])
PLAIN = np.array([225, 215, 200])
BLUE = np.array([60, 125, 235])
IVORY = np.array([250, 245, 225])
PALETTE = (np.array(plt.get_cmap("tab20").colors) * 255).astype(int)


def tooth_color(fdi: int) -> np.ndarray:
    q, p = divmod(int(fdi), 10)
    return PALETTE[(2 * (p - 1) + (0 if q in (1, 4) else 1)) % 20]


def label_colors(labels: np.ndarray) -> np.ndarray:
    out = np.tile(GUM, (len(labels), 1))
    for f in np.unique(labels):
        if f > 0:
            out[labels == f] = tooth_color(f)
    return out


def sample(mesh: trimesh.Trimesh, vertex_colors: np.ndarray, per_mm2: float = 260.0, seed: int = 0):
    """表面采样点、颜色（取所在三角形第一个顶点的颜色）、法向。"""
    n = int(min(3_000_000, max(20_000, mesh.area * per_mm2)))
    pts, face = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts), vertex_colors[mesh.faces[face, 0]], mesh.face_normals[face]


def camera(up: np.ndarray, toward: np.ndarray, elevation_deg: float):
    """相机在“水平方向 toward、仰角 elevation”的方向上看向场景。返回 (right, upv, d)。"""
    up = up / np.linalg.norm(up)
    h = toward - (toward @ up) * up
    h = h / np.linalg.norm(h)
    e = np.deg2rad(elevation_deg)
    d = np.cos(e) * h + np.sin(e) * up
    right = np.cross(up, d) if elevation_deg < 89 else np.cross(up, h)
    right = right / np.linalg.norm(right)
    return right, np.cross(d, right), d


def render(parts, cam, center, half: float, size: int = 520, aspect: float = 1.0) -> np.ndarray:
    """parts: [(点, 颜色, 法向), …]。正交投影，画布宽 2·half mm。"""
    right, upv, d = cam
    w, h = size, int(size * aspect)
    img = np.full((h, w, 3), 255, np.uint8)
    pts = np.concatenate([p for p, _, _ in parts]) - center
    col = np.concatenate([c for _, c, _ in parts]).astype(float)
    nrm = np.concatenate([n for _, _, n in parts])
    shade = 0.35 + 0.65 * np.abs(nrm @ d)
    col = np.clip(col * shade[:, None], 0, 255).astype(np.uint8)
    scale = w / (2 * half)
    x = (pts @ right) * scale + w / 2
    y = h / 2 - (pts @ upv) * scale
    order = np.argsort(pts @ d)  # 远的先画
    x, y, col = x[order], y[order], col[order]
    for dx in (0, 1):
        for dy in (0, 1):
            xi, yi = np.floor(x).astype(int) + dx, np.floor(y).astype(int) + dy
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            img[yi[ok], xi[ok]] = col[ok]
    return img


def line_part(points: np.ndarray, color, d: np.ndarray, lift: float = 0.25, thick: float = 0.07):
    """折线画成一串小点，朝相机方向抬一点避免被表面挡住。"""
    seg = np.roll(points, -1, axis=0) - points
    n = np.maximum(1, (np.linalg.norm(seg, axis=1) / 0.02).astype(int))
    dense = np.concatenate([points[i] + np.linspace(0, 1, k, endpoint=False)[:, None] * seg[i] for i, k in enumerate(n)])
    rng = np.random.default_rng(0)
    dense = np.repeat(dense, 6, axis=0) + rng.normal(0, thick, (len(dense) * 6, 3)) + d * lift
    return dense, np.tile(np.array(color), (len(dense), 1)), np.tile(d, (len(dense), 1))


def grid(rows, col_titles, row_titles, path: Path, panel: float = 3.0):
    fig, axes = plt.subplots(len(rows), len(rows[0]), figsize=(panel * len(rows[0]), panel * len(rows) * rows[0][0].shape[0] / rows[0][0].shape[1] + 0.35),
                             squeeze=False)
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            ax = axes[i][j]
            ax.imshow(im)
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#cccccc")
            if i == 0:
                ax.set_title(col_titles[j], fontsize=11)
            if j == 0 and row_titles:
                ax.set_ylabel(row_titles[i], fontsize=10)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("写入", path, f"{path.stat().st_size / 1024:.0f} KB", flush=True)


def scan(name: str):
    case = name.rsplit("_", 1)[0]
    obj = ROOT / "data" / "fair_test" / case / f"{name}.obj"
    mesh = load_mesh(obj)
    labels, jaw = load_labels(obj.with_suffix(".json"))
    return mesh, labels, jaw


def occlusal(mesh, labels, jaw):
    arch = estimate_arch_frame(mesh, labels, jaw)
    up = arch.occlusal
    front = np.cross(arch.lateral, up) * (1 if jaw == "lower" else -1)
    cam = camera(up, -front, 89.9)
    center = mesh.vertices[labels > 0].mean(axis=0)
    return cam, center


# ---------------------------------------------------------------- 各张图


def fig_segmentation(seg, out: Path):
    from dental_ai.segmentation.numbering import correct_numbering
    from dental_ai.segmentation.postprocess import remove_small_islands

    rows, titles = [], []
    for name in ("0AAQ6BO3_upper", "09HRNJZ8_lower"):
        mesh, gt, jaw = scan(name)
        pred = remove_small_islands(mesh, correct_numbering(mesh, seg.predict(mesh, jaw), jaw).labels)
        cam, center = occlusal(mesh, gt, jaw)
        acc = float((pred == gt).mean())
        ims = [render([sample(mesh, c)], cam, center, 36, aspect=0.85)
               for c in (np.tile(PLAIN, (len(gt), 1)), label_colors(pred), label_colors(gt))]
        rows.append(ims)
        titles.append(f"{jaw} jaw\nvertex acc {acc:.1%}")
    grid(rows, ["input scan", "model prediction", "ground truth"], titles, out / "segmentation.png")


def fig_missing_seg(seg_old, seg_new, out: Path):
    from dental_ai.segmentation.numbering import correct_numbering
    from dental_ai.segmentation.postprocess import remove_small_islands

    before = {(r["scan"], r["scenario"]): r for r in json.loads((ROOT / "runs/missing_seg/deep_before/rows.json").read_text())}
    after = json.loads((ROOT / "runs/seg_deep_attn/missing/rows.json").read_text())
    # 旧模型出现幻影牙、新模型完全正确的病例里取前两个（按扫描名排序，不看图挑）
    picks = [r for r in sorted(after, key=lambda r: r["scan"]) if r["scenario"] == "multi" and r.get("exact")
             and all(before[(r["scan"], "multi")].get("phantom", [False]))][:2]
    rows, titles = [], []
    for r in picks:
        mesh0, gt0, jaw = scan(r["scan"])
        mesh, gt = simulate_missing(mesh0, gt0, jaw, r["teeth"])
        cam, center = occlusal(mesh0, gt0, jaw)
        preds = [remove_small_islands(mesh, correct_numbering(mesh, s.predict(mesh, jaw), jaw).labels) for s in (seg_old, seg_new)]
        ims = [render([sample(mesh, c)], cam, center, 36, aspect=0.85)
               for c in (np.tile(PLAIN, (len(gt), 1)), label_colors(preds[0]), label_colors(preds[1]), label_colors(gt))]
        rows.append(ims)
        titles.append(f"{jaw}, removed {' '.join(map(str, r['teeth']))}")
    grid(rows, ["scan with teeth removed", "old model (phantom teeth)", "new model", "ground truth"], titles, out / "segmentation_missing.png", panel=2.8)


def fig_generation(out: Path):
    from dental_ai.eval_generation import _Fixed
    from dental_ai.generation.learned import make_generator
    from dental_ai.pipeline import run

    rows_csv = [r for r in csv.DictReader(open(ROOT / "runs/gen_compare/learned/generation.csv")) if r.get("chamfer_l1")]
    mirror = {(r["scan"], r["scenario"], r["fdi"]): r for r in csv.DictReader(open(ROOT / "runs/gen_compare/mirror/generation.csv")) if r.get("chamfer_l1")}

    def pick(scenario, q, cond=lambda r: True):
        sub = sorted((r for r in rows_csv if r["scenario"] == scenario and cond(r)), key=lambda r: float(r["chamfer_l1"]))
        return sub[int(q * (len(sub) - 1))]

    picks = [("single gap, premolar\n(median case)", pick("single", 0.5, lambda r: int(r["fdi"]) % 10 in (4, 5))),
             ("single gap, molar\n(median case)", pick("single", 0.5, lambda r: int(r["fdi"]) % 10 == 6)),
             ("free end, 2nd molar\n(median case)", pick("free_end", 0.5)),
             ("single gap\n(90th percentile = poor)", pick("single", 0.9))]
    gens = {"learned": make_generator(str(ROOT / "models/generation/toothgen_open.pt")), "mirror": make_generator(None)}
    rows, titles, notes = [], [], []
    for title, r in picks:
        fdi = int(r["fdi"])
        mesh0, gt0, jaw = scan(r["scan"])
        teeth = tuple(int(t) for t in r["teeth"].split())
        mesh, labels = simulate_missing(mesh0, gt0, jaw, teeth)
        arch = estimate_arch_frame(mesh0, gt0, jaw)
        up = arch.occlusal
        real = tooth_mesh(mesh0, gt0, fdi)
        center = real.vertices.mean(axis=0)
        radial = center - mesh0.vertices[gt0 > 0].mean(axis=0)
        cam = camera(up, radial, 38)
        base = sample(mesh, np.tile(GUM, (len(labels), 1)) * (labels[:, None] == 0) + np.tile(IVORY, (len(labels), 1)) * (labels[:, None] > 0), per_mm2=500)
        ims = [render([base], cam, center, 13)]
        for name in ("mirror", "learned"):
            tmp = Path(tempfile.mkdtemp())
            rep = run(mesh, jaw, _Fixed(labels), tmp, generator=gens[name])
            g = next(x for x in rep["generated"] if x["fdi"] == fdi)
            solid = trimesh.load(tmp / g["file"])
            ims.append(render([base, sample(solid, np.tile(BLUE, (len(solid.vertices), 1)), per_mm2=500)], cam, center, 13))
        full = sample(mesh0, np.tile(GUM, (len(gt0), 1)) * (gt0[:, None] == 0) + np.tile(IVORY, (len(gt0), 1)) * (gt0[:, None] > 0), per_mm2=500)
        ims.append(render([full], cam, center, 13))
        rows.append(ims)
        m = mirror.get((r["scan"], r["scenario"], r["fdi"]))
        titles.append(f"{title}\ntooth {fdi}")
        notes.append({"scan": r["scan"], "fdi": fdi, "scenario": r["scenario"], "learned_chamfer_mm": round(float(r["chamfer_l1"]), 3),
                      "mirror_chamfer_mm": round(float(m["chamfer_l1"]), 3) if m else None, "mirror_source": m["source"] if m else None})
    grid(rows, ["missing tooth", "geometric method (blue)", "learned model (blue)", "real tooth"], titles, out / "generation.png", panel=2.7)
    (out / "generation_cases.json").write_text(json.dumps(notes, indent=2))
    print(json.dumps(notes, indent=1))


def fig_margin(out: Path):
    import torch

    from dental_ai.eval_margin import load_vtp, truth_line
    from dental_ai.margin import learned as ml

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ml._model().to(device)
    model.load_state_dict(torch.load(ROOT / "models/margin/margin.pt", map_location=device)["state"])
    cases = {c["case"]: c for c in json.loads((ROOT / "runs/margin/eval_test/cases.json").read_text())}
    order = sorted(cases.values(), key=lambda c: c["learned_mean"])
    picks = [order[len(order) // 2], order[-1]]  # 中位数病例和最差病例
    ims, titles = [], []
    for c in picks:
        path = next((ROOT / "data/prep_margin/test Data").glob(f"{c['case']}*.vtp"))
        mesh, face_label = load_vtp(path)
        line = truth_line(mesh, face_label, spacing=0.05)
        pts, field = ml.detect(model, mesh, device)
        up = mesh.vertex_normals[field > 0].sum(axis=0)
        up /= np.linalg.norm(up)
        center = line.mean(axis=0)
        side = np.cross(up, [1.0, 0.0, 0.0])
        for toward in (side, -side):
            cam = camera(up, toward, 30)
            surf = sample(mesh, np.tile(PLAIN, (len(mesh.vertices), 1)), per_mm2=900)
            parts = [surf, line_part(line, (40, 170, 60), cam[2], lift=0.30, thick=0.06), line_part(pts, (225, 40, 40), cam[2], lift=0.45, thick=0.035)]
            ims.append(render(parts, cam, center, 7.5))
        titles += [f"error {c['learned_mean']:.3f} mm", "same case, other side"]
    titles[0] = "median case: " + titles[0]
    titles[2] = "worst case: " + titles[2]
    grid([ims], titles, None, out / "margin.png", panel=3.0)


def fig_charts(out: Path):
    S = {n: json.loads((ROOT / f"runs/gen_compare/{n}/generation_summary.json").read_text()) for n in ("mirror", "learned")}
    scen = ["single", "both_sides", "free_end", "free_end_both"]
    names = ["single gap", "both sides\nmissing", "free end", "free end,\nboth sides"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    x = np.arange(len(scen))
    for k, (n, lab, col) in enumerate((("mirror", "geometric (mirror / template)", "#b0b0b0"), ("learned", "learned model", "#3c7deb"))):
        v = [S[n][s]["chamfer_l1_median"] for s in scen]
        b = ax.bar(x + (k - 0.5) * 0.38, v, 0.38, label=lab, color=col)
        ax.bar_label(b, fmt="%.2f", fontsize=9)
    ax.set_xticks(x, names), ax.set_ylabel("median Chamfer distance to real tooth (mm)"), ax.set_title("Missing-tooth generation (lower is better)")
    ax.legend(frameon=False), ax.spines[["top", "right"]].set_visible(False)

    M = {"old model": json.loads((ROOT / "runs/missing_seg/deep_before/summary.json").read_text()),
         "DGCNN + simulated gaps": json.loads((ROOT / "runs/seg_deep_missing/missing/summary.json").read_text()),
         "DGCNN-attn + simulated gaps": json.loads((ROOT / "runs/seg_deep_attn/missing/summary.json").read_text())}
    ax = axes[1]
    keys = [("one", "phantom"), ("multi", "phantom"), ("one", "detected"), ("multi", "detected")]
    names = ["phantom teeth\n(1 removed)", "phantom teeth\n(2-3 removed)", "gap detected\n(1 removed)", "gap detected\n(2-3 removed)"]
    x = np.arange(len(keys))
    for k, (lab, col) in enumerate(zip(M, ("#b0b0b0", "#8fb4f0", "#3c7deb"))):
        v = [100 * M[lab][a][b] for a, b in keys]
        b = ax.bar(x + (k - 1) * 0.27, v, 0.27, label=lab, color=col)
        ax.bar_label(b, fmt="%.0f", fontsize=8)
    ax.set_xticks(x, names, fontsize=8), ax.set_ylabel("% of removed teeth"), ax.set_title("Segmentation on scans with missing teeth\n(phantom: lower is better; detected: higher is better)", fontsize=11)
    ax.legend(frameon=False, fontsize=8), ax.spines[["top", "right"]].set_visible(False), ax.set_ylim(0, 105)

    ax = axes[2]
    E = {"old model": "deep", "DGCNN + simulated gaps": "seg_deep_missing", "DGCNN-attn + simulated gaps": "seg_deep_attn"}
    keys = [("eval_fair_test", "score"), ("eval_fair_test", "t_miou"), ("eval_test-complex", "score"), ("eval_test-complex", "t_miou")]
    names = ["standard test\nscore", "standard test\ntooth mIoU", "hard cases\nscore", "hard cases\ntooth mIoU"]
    x = np.arange(len(keys))
    for k, (lab, col) in enumerate(zip(E, ("#b0b0b0", "#8fb4f0", "#3c7deb"))):
        v = [100 * json.loads((ROOT / f"runs/{E[lab]}/{a}/summary.json").read_text())["all"][b] for a, b in keys]
        b = ax.bar(x + (k - 1) * 0.27, v, 0.27, label=lab, color=col)
        ax.bar_label(b, fmt="%.1f", fontsize=8)
    ax.set_xticks(x, names, fontsize=8), ax.set_ylabel("%"), ax.set_title("Segmentation accuracy (higher is better)"), ax.set_ylim(40, 102)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "metrics.png", dpi=110)
    plt.close(fig)
    print("写入", out / "metrics.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--old-seg", type=Path, help="只用完整牙列训练的旧分割模型（“修复前”的对比）；不给就不画那张图")
    p.add_argument("--only", nargs="*", default=[])
    args = p.parse_args()
    out = ROOT / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    from dental_ai.segmentation.deep import DeepSegmenter

    todo = set(args.only) or {"charts", "segmentation", "missing", "generation", "margin"}
    if "charts" in todo:
        fig_charts(out)
    seg = DeepSegmenter(ROOT / "models/segmentation/deep.pt") if todo & {"segmentation", "missing"} else None
    if "segmentation" in todo:
        fig_segmentation(seg, out)
    if "missing" in todo and args.old_seg:
        fig_missing_seg(DeepSegmenter(args.old_seg), seg, out)
    if "generation" in todo:
        fig_generation(out)
    if "margin" in todo:
        fig_margin(out)


if __name__ == "__main__":
    main()
