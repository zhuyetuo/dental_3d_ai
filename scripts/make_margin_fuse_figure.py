"""颈缘线融合（11-25）的效果图：口扫上的模拟预备体，真值线 / 几何方法 / 融合 并排；加一张逐牙误差分布图。
用 scripts/margin_fuse_eval.py 的结果（runs/margin_fuse/rows.json）挑病例。"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "scripts"))
from make_result_figures import GUM, camera, grid, line_part, render, sample  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

GATE = 0.8
ZIP = Path("/home/zyt/zyt/dental_data/Bite2Text.zip")
MODEL = ROOT / "models/margin/margin_scan_real.pt"


def fused_err(r):
    return r["snap0.5"] if "learned" in r and r["dist_lg"] < GATE else r["geo"]


def main():
    from dental_ai.cache import cached_preparation
    from dental_ai.eval_crown import _labels, _load
    from dental_ai.geometry import estimate_arch_frame
    from dental_ai.margin.detect import detect_margin, edge_sharpness
    from dental_ai.margin.learned import detect_in_scan
    from dental_ai.synthetic import REAL_PREP_KW

    out = ROOT / "docs/images"
    rows = [r for r in json.loads((ROOT / "runs/margin_fuse/rows.json").read_text()) if "geo" in r]
    for r in rows:
        r["fused"] = fused_err(r)
        r["gain"] = r["geo"] - r["fused"]
    used = sorted([r for r in rows if "learned" in r and r["dist_lg"] < GATE], key=lambda r: r["gain"])
    kept = sorted([r for r in rows if not ("learned" in r and r["dist_lg"] < GATE)], key=lambda r: r["geo"])
    picks = [used[-2], used[len(used) // 2], used[0]]
    tags = ["big gain", "typical", "worst for fused"]

    ims, row_titles = [], []
    for r, tag in zip(picks, tags):
        mesh = _load(ZIP, r["patient"], r["jaw"])
        lab = _labels(mesh, r["patient"], r["jaw"], str(ROOT / "models/segmentation/deep.pt"), ROOT / "runs/margin_fuse/labels")
        up = estimate_arch_frame(mesh, lab, r["jaw"]).occlusal
        fdi = r["fdi"]
        prep, truth, _ = cached_preparation(mesh, lab, r["jaw"], fdi, **REAL_PREP_KW)
        sharp = edge_sharpness(prep)
        geo = detect_margin(prep, lab, fdi, up=up, sharpness=sharp).points
        try:
            L = detect_in_scan(str(MODEL), prep, lab, fdi)
            d = (cKDTree(geo).query(L)[0].mean() + cKDTree(L).query(geo)[0].mean()) / 2
            fused = detect_margin(prep, lab, fdi, up=up, sharpness=sharp, initial=L, search_radius=0.5, max_height_diff=0.5).points if d < GATE else geo
        except ValueError:
            fused = geo
        v = np.asarray(prep.vertices)
        c = truth.mean(axis=0)
        keep = np.linalg.norm(v - c, axis=1) < 9.0
        crop = prep.submesh([np.flatnonzero(keep[prep.faces].all(axis=1))], append=True)
        cl = lab[keep]  # submesh 保留顶点顺序：用最近点取标签更稳
        cl = lab[cKDTree(v).query(np.asarray(crop.vertices))[1]]
        col = np.tile(GUM, (len(crop.vertices), 1))
        col[cl == fdi] = (238, 232, 212)
        col[(cl > 0) & (cl != fdi)] = (205, 200, 190)
        toward = np.cross(up, [0.0, 0.0, 1.0]) if abs(up[2]) < 0.9 else np.cross(up, [1.0, 0.0, 0.0])
        # 朝颊侧：从牙弓中心指向这颗牙
        hv = c - np.asarray(mesh.vertices).mean(axis=0)
        hv -= (hv @ up) * up
        if np.linalg.norm(hv) > 1:
            toward = hv / np.linalg.norm(hv)
        cam = camera(up, toward, 60)
        surf = sample(crop, col, per_mm2=700)
        # 真值是没连起来的顶点（肩台外缘 + 邻面），画成点，不连线（按角度连会来回折）
        tp = np.repeat(truth, 16, axis=0) + np.random.default_rng(0).normal(0, 0.08, (len(truth) * 16, 3)) + cam[2] * 0.6
        tpart = (tp, np.tile((40, 170, 60), (len(tp), 1)), np.tile(cam[2], (len(tp), 1)))
        row = []
        for line, rgb in ((geo, (40, 110, 230)), (fused, (225, 40, 40))):
            parts = [surf, tpart, line_part(line, rgb, cam[2], lift=0.45, thick=0.035)]
            row.append(render(parts, cam, c + up * 1.0, 7.0, size=460))
        ims.append(row)
        row_titles.append(f"{tag}\n{r['geo']:.2f} -> {r['fused']:.2f} mm")
    grid(ims, ["geometric (blue)", "fused (red)"], row_titles, out / "margin_fuse.png", panel=3.4)

    # 逐牙误差分布（累计）
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for key, name, c in (("geo", "geometric (online)", "#2e6fe0"), ("learned", "learned only", "#999999"), ("fused", "fused", "#d62728")):
        e = np.sort([r.get(key, r["geo"]) for r in rows])
        ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=c, lw=2.2 if key == "fused" else 1.6, label=f"{name}  mean {np.mean(e):.2f} mm")
    ax.axvline(0.2, color="#bbbbbb", ls="--", lw=1)
    ax.set_xlim(0, 1.6)
    ax.set_xlabel("margin line error per tooth (mm)")
    ax.set_ylabel("share of teeth")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "margin_fuse_cdf.png", dpi=130)
    print("写入", out / "margin_fuse_cdf.png")


if __name__ == "__main__":
    main()
