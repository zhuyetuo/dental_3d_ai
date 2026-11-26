"""体素模型第二版（咬合面加权 + 多尺度监督）的报告图：
docs/images/voxel2_occlusal.png：咬合面俯视，技师牙冠 / 旧版 / 新版，颜色 = 离技师牙冠的距离，圈 = 技师牙冠的牙尖，叉 = 生成牙冠的牙尖；
docs/images/voxel2_bars.png：牙尖 / 窝误差（150 例）和 CD-L2（300 例）。"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path.home() / "zyt/dental_3d_ai/ml"))
from dental_ai.crown_quality import occlusal_landmarks  # noqa: E402
from dental_ai.eval_crownfusion import surface_points  # noqa: E402

plt.rcParams["font.family"] = ["Noto Sans CJK SC", "Noto Serif CJK SC", "DejaVu Sans"]
ROOT = Path.home() / "zyt/dental_3d_ai"
A, B = ROOT / "runs/voxel2/mesh_voxel", ROOT / "runs/voxel2/mesh_voxel2"
OUT = ROOT / "docs/images"
UP = np.array([0, 1.0, 0])
VMAX = 0.6


def marks(m):
    return occlusal_landmarks(surface_points(m, 12000, 3), UP)


def cusp_err(m, g):
    k = marks(m)
    return float(cKDTree(k["cusps"]).query(g["cusps"])[0].mean()), k


rows = []
for f in sorted(A.glob("*_gt.ply")):
    cid = f.name.split("_")[0]
    gt = trimesh.load(f, process=False)
    v1 = trimesh.load(A / f"{cid}_vox.ply", process=False)
    v2 = trimesh.load(B / f"{cid}_vox.ply", process=False)
    try:
        g = marks(gt)
        e1, k1 = cusp_err(v1, g)
        e2, k2 = cusp_err(v2, g)
    except Exception as exc:  # noqa: BLE001
        print("skip", cid, exc)
        continue
    rows.append(dict(cid=cid, gt=gt, v1=v1, v2=v2, g=g, k1=k1, k2=k2, e1=e1, e2=e2))
    print(cid, round(e1, 3), round(e2, 3))
rows.sort(key=lambda r: r["e2"] - r["e1"])
pick = rows[:3] + rows[-1:]  # 改善最多的 3 例 + 变差最多的 1 例


def draw(ax, m, gt_tree, gmarks, pmarks, title, color_by_dist=True):
    v, fc = np.asarray(m.vertices), np.asarray(m.faces)
    tri = v[fc]
    top = tri[:, :, 1].mean(1)
    keep = top > np.percentile(v[:, 1], 25)
    o = np.argsort(top[keep])
    tri = tri[keep][o]
    n = np.asarray(m.face_normals)[keep][o]
    shade = 0.45 + 0.55 * np.clip(np.abs(n[:, 1]) * 0.7 + 0.3 * np.clip(n[:, 0], 0, 1), 0, 1)
    if color_by_dist:
        d = gt_tree.query(tri.mean(1))[0]
        col = plt.cm.magma_r(np.clip(d / VMAX, 0, 1) * 0.85)[:, :3]
    else:
        col = np.tile([0.86, 0.84, 0.78], (len(tri), 1))
    col = col * shade[:, None]
    ax.add_collection(PolyCollection(tri[:, :, [0, 2]], facecolors=col, edgecolors="none"))
    gc = gmarks["cusps"]
    ax.scatter(gc[:, 0], gc[:, 2], s=70, facecolors="none", edgecolors="#2bb3c0", linewidths=2, zorder=5)
    if pmarks is not None:
        pc = pmarks["cusps"]
        ax.scatter(pc[:, 0], pc[:, 2], s=60, marker="x", c="#111", linewidths=2, zorder=6)
        j = cKDTree(pc).query(gc)[1]
        for a_, b_ in zip(gc, pc[j]):
            ax.plot([a_[0], b_[0]], [a_[2], b_[2]], c="#2bb3c0", lw=1.2, zorder=4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.autoscale_view()
    ax.set_title(title, fontsize=11)


fig, axs = plt.subplots(len(pick), 3, figsize=(10.5, 3.4 * len(pick)))
for i, r in enumerate(pick):
    tree = cKDTree(surface_points(r["gt"], 40000, 9))
    tag = "改善最多" if i < 3 else "变差最多"
    draw(axs[i, 0], r["gt"], tree, r["g"], None, f"病例 {r['cid']}（{tag}）· 技师牙冠", False)
    draw(axs[i, 1], r["v1"], tree, r["g"], r["k1"], f"旧版 · 牙尖误差 {r['e1']:.2f} mm")
    draw(axs[i, 2], r["v2"], tree, r["g"], r["k2"], f"新版 · 牙尖误差 {r['e2']:.2f} mm")
sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(plt.cm.magma_r(np.linspace(0, 0.85, 256))),
                           norm=matplotlib.colors.Normalize(0, VMAX))
cb = fig.colorbar(sm, ax=axs, orientation="horizontal", fraction=0.02, pad=0.02, aspect=50)
cb.set_label("离技师牙冠的距离（mm）　○ 技师牙冠的牙尖　× 生成牙冠的牙尖")
fig.savefig(OUT / "voxel2_occlusal.png", dpi=110, bbox_inches="tight")

# 柱状图
fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
data = [("牙尖误差（mm，150 例）", [0.884, 0.801], [0.653, 0.582]),
        ("中央窝误差（mm，150 例）", [1.102, 1.084], [0.497, 0.483]),
        ("网格 CD-L2（mm，300 例）", [0.334, 0.336], [0.310, 0.315])]
for ax, (t, mean, med) in zip(axs, data):
    x = np.arange(2)
    ax.bar(x - 0.18, mean, 0.36, color=["#b9b4a8", "#d0573a"], label="平均")
    ax.bar(x + 0.18, med, 0.36, color=["#dcd8cf", "#eea48f"], label="中位")
    for xi, a_, b_ in zip(x, mean, med):
        ax.text(xi - 0.18, a_, f"{a_:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(xi + 0.18, b_, f"{b_:.3f}", ha="center", va="bottom", fontsize=9)
    if "CD-L2" in t:
        ax.axhline(0.353, ls="--", c="#555", lw=1); ax.text(1.45, 0.353, "VBCD 0.353", fontsize=9, va="bottom", ha="right")
        ax.axhline(0.412, ls=":", c="#555", lw=1); ax.text(1.45, 0.412, "CrownFusion 0.412", fontsize=9, va="bottom", ha="right")
        ax.set_ylim(0, 0.47)
    ax.set_xticks(x, ["旧版", "新版"]); ax.set_title(t, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "voxel2_bars.png", dpi=120)
print("FIG_DONE", [r["cid"] for r in pick])
