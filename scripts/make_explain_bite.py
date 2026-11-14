"""docs/images/explain_bite.png：“和对颌”指标的真实剖面图（扁平配色）。

Bite2Text 里一颗真的下颌磨牙和它上方的对颌，沿颊舌方向（垂直于近远中）切一刀，取咬合那一侧的真实轮廓（牙冠的上边界、对颌的下边界）。
三个面板：把牙冠沿咬合方向挪动，剖面上分别是“没咬上（离开 0.6 mm）”“刚好接触”“穿进对颌 0.5 mm”。

    python scripts/make_explain_bite.py [--patient F5075 --fdi 36]
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == "scripts" else Path.home() / "zyt/dental_3d_ai"
sys.path.insert(0, str(ROOT / "ml"))
from dental_ai.crown_quality import _crop  # noqa: E402
from dental_ai.eval_crown import _labels, _load  # noqa: E402
from dental_ai.geometry import estimate_arch_frame, tooth_centroids, tooth_mesh  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Serif CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
CROWN, CROWN_E, ANT, ANT_E = "#33c3d1", "#0f7d89", "#aab8d3", "#6c7c9f"
BAD, GOOD, WARN, BG = "#ff5a5a", "#2fd08a", "#f5b74f", "#0b1426"
GRID = np.linspace(-6.5, 6.5, 521)


def envelope(mesh, origin, normal, u, v, upper: bool):
    """平面切开网格，取每一列（u）上 v 的最大值（upper，牙冠的咬合面）或最小值（对颌的咬合面）。"""
    import trimesh

    out = np.full(len(GRID), np.nan)
    segs = trimesh.intersections.mesh_plane(mesh, plane_normal=normal, plane_origin=origin)  # (N, 2, 3) 线段
    if len(segs) == 0:
        return out
    t = np.linspace(0, 1, 6)[None, :, None]  # 每段加密，列里不会漏
    p = (segs[:, :1] + t * (segs[:, 1:] - segs[:, :1])).reshape(-1, 3) - origin
    pu, pv = p @ u, p @ v
    idx = np.clip(np.round((pu - GRID[0]) / (GRID[1] - GRID[0])).astype(int), 0, len(GRID) - 1)
    ok = (pu >= GRID[0]) & (pu <= GRID[-1])
    for i, val in zip(idx[ok], pv[ok]):
        if np.isnan(out[i]) or (val > out[i] if upper else val < out[i]):
            out[i] = val
    good = ~np.isnan(out)
    if good.sum() >= 2:  # 中间零星没点的列用两边插值补上（不然填色会出现竖缝），两端不外推
        lo, hi = np.flatnonzero(good)[[0, -1]]
        inside = np.arange(lo, hi + 1)
        out[inside] = np.interp(inside, np.flatnonzero(good), out[good])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(Path.home() / "zyt/dental_data/Bite2Text.zip"))
    ap.add_argument("--patient", default="F5075")
    ap.add_argument("--fdi", type=int, default=36)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/images/explain_bite.png")
    a = ap.parse_args()
    jaw = "lower" if a.fdi // 10 in (3, 4) else "upper"
    other = "upper" if jaw == "lower" else "lower"
    mesh = _load(a.zip, a.patient, jaw)
    lab = _labels(mesh, a.patient, jaw, str(ROOT / "models/segmentation/deep.pt"), ROOT / "runs/crown_quality/labels")
    v = estimate_arch_frame(mesh, lab, jaw).occlusal
    v = v / np.linalg.norm(v)
    crown = tooth_mesh(mesh, lab, a.fdi)
    ant = _crop(_load(a.zip, a.patient, other), np.asarray(crown.vertices), 10.0)
    cents = tooth_centroids(mesh, lab)
    nb = [cents[f] for f in (a.fdi - 1, a.fdi + 1) if f in cents]
    md = (nb[-1] - nb[0]) if len(nb) == 2 else np.cross(v, [1.0, 0, 0])
    md = md - v * (md @ v)
    md /= np.linalg.norm(md)  # 近远中方向：剖面的法向
    u = np.cross(md, v)
    center = np.asarray(crown.vertices).mean(axis=0)
    c_env = envelope(crown, center, md, u, v, upper=True)
    a_env = envelope(ant, center, md, u, v, upper=False)
    both = ~np.isnan(c_env) & ~np.isnan(a_env)
    gap0 = float(np.nanmin((a_env - c_env)[both]))  # 剖面上最窄处的间隙
    base = float(np.nanmin(c_env)) - 1.0
    cases = [("没咬上：离开 0.6 mm", gap0 - 0.6, WARN), ("刚好接触 ✓", gap0, GOOD), ("穿进对颌 0.5 mm", gap0 + 0.5, BAD)]
    top = float(np.nanmax(a_env)) + 1.0
    lo, hi = base + 0.6, top - 0.2

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor=BG)
    for ax, (title, shift, color) in zip(axes, cases):
        ax.set_facecolor(BG)
        c = c_env + shift
        m = ~np.isnan(c)
        ax.fill_between(GRID, top, a_env, where=~np.isnan(a_env), color=ANT, lw=0)
        ax.plot(GRID, a_env, color=ANT_E, lw=2.2)
        ax.fill_between(GRID, base + shift, c, where=m, color=CROWN, lw=0)
        ax.plot(GRID, c, color=CROWN_E, lw=2.2)
        diff = a_env - c
        over = both & (diff < 0)
        if over.any():
            ax.fill_between(GRID, a_env, c, where=over, color=BAD, lw=0, alpha=0.95)
        touch = both & (np.abs(diff) <= 0.06)
        if touch.any() and not over.any():
            ax.scatter(GRID[touch][::6], c[touch][::6], s=70, color=GOOD, edgecolor="white", linewidth=1.2, zorder=5)
        if shift < gap0:  # 标出最窄处的间隙
            i = int(np.nanargmin(np.where(both, diff, np.nan)))
            ax.annotate("", xy=(GRID[i], a_env[i]), xytext=(GRID[i], c[i]), arrowprops=dict(arrowstyle="<|-|>", color=WARN, lw=3, mutation_scale=16))
            ax.text(GRID[i] + 0.35, (a_env[i] + c[i]) / 2, "空 0.6 mm", color=WARN, fontsize=15, fontweight="bold", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="none"))
        ax.set_xlim(GRID[0], GRID[-1])
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, color=color, fontsize=17, fontweight="bold", pad=8)
    axes[0].text(GRID[0] + 0.2, hi - 0.6, "对颌（上面的牙）", color="#1d2a44", fontsize=12, fontweight="bold")
    for ax in axes:
        ax.text(0, lo + 0.25, "牙冠", color="#06343a", fontsize=13, fontweight="bold", ha="center")
    fig.text(0.5, 0.03, f"真实剖面：Bite2Text {a.patient} 的 {a.fdi} 号牙和对颌，沿颊舌方向切开（单位 mm）。绿点 = 刚好接触；红色 = 穿进对颌的部分",
             color="#9aa8c4", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(a.out, dpi=150, facecolor=BG)
    print("saved", a.out, "gap0", round(gap0, 3))
    export_interactive(a, crown, ant, center, u, v, md, c_env, a_env, gap0)


def export_interactive(a, crown, ant, center, u, v, md, c_env, a_env, gap0):
    """网页上的“动图”和“3D 交互”用的数据：剖面轮廓（JSON）+ 牙冠和对颌网格（二进制），坐标转到 x=颊舌、y=咬合方向、z=近远中。"""
    import json

    from dental_ai.crown_quality import _crop
    from dental_ai.generation.contacts import signed_distance_to_surface

    out = ROOT / "docs/assets/bite"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{a.patient}_{a.fdi}"
    R = np.stack([u, v, md])  # 行向量：新坐标轴

    def local(p):
        return ((np.asarray(p) - center) @ R.T).astype(np.float32)

    ant_small = _crop(ant, np.asarray(crown.vertices), 7.0)
    d, _ = signed_distance_to_surface(np.asarray(crown.vertices), ant_small, cutoff=4.0)
    d = np.where(np.isfinite(d), d, 4.0).astype(np.float32)
    meta = {"grid": np.round(GRID, 3).tolist(), "crown": [None if np.isnan(x) else round(float(x), 3) for x in c_env],
            "ant": [None if np.isnan(x) else round(float(x), 3) for x in a_env], "gap0": round(gap0, 3),
            "patient": a.patient, "fdi": a.fdi, "contact_shift": round(float(np.min(d)), 3),
            "crown_verts": len(crown.vertices), "crown_faces": len(crown.faces), "ant_verts": len(ant_small.vertices), "ant_faces": len(ant_small.faces)}
    (out / f"{stem}.json").write_text(json.dumps(meta))
    with open(out / f"{stem}.bin", "wb") as f:  # 顺序：牙冠顶点、牙冠面、牙冠到对颌的距离、对颌顶点、对颌面
        f.write(local(crown.vertices).tobytes())
        f.write(np.asarray(crown.faces, np.uint32).tobytes())
        f.write(d.tobytes())
        f.write(local(ant_small.vertices).tobytes())
        f.write(np.asarray(ant_small.faces, np.uint32).tobytes())
    print("interactive data", out / f"{stem}.bin", round((out / f"{stem}.bin").stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
