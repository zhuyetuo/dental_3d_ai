"""真实预备体测试集咬合面（斜看）：技师牙冠 / 形状模型 / 加扩散细节（采样 1 次）/ 加扩散细节（采样 8 次取平均）。"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import trimesh
from matplotlib import font_manager

sys.path.insert(0, str(Path.home() / "zyt/dental_3d_ai/scripts"))
sys.path.insert(0, str(Path.home() / "zyt/dental_3d_ai/ml"))
import make_result_figures as F  # noqa: E402
from dental_ai.generation.radial_crownfusion import R_CF  # noqa: E402

for f in font_manager.findSystemFonts():
    if any(k in f for k in ("NotoSansCJK", "NotoSansSC", "wqy", "SourceHanSans", "DroidSansFallback", "simhei")):
        font_manager.fontManager.addfont(f)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break

D = Path.home() / "zyt/dental_data/crownfusion/CrownFusion/test"
ONE, AVG = Path(sys.argv[1]), Path(sys.argv[2])
out = Path(sys.argv[3])
rows_avg = {r["case"]: r for r in json.load(open(AVG / "rows.json"))}
cases = sorted({p.name.split("_")[0] for p in AVG.glob("*_diff.ply")} & {p.name.split("_")[0] for p in ONE.glob("*_diff.ply")})
med = np.median([r["diff_cd"] for r in rows_avg.values()])
cases = sorted(cases, key=lambda c: abs(rows_avg[c]["diff_cd"] - med))[:5]
GT, DC, D1, D8 = [235, 205, 120], [150, 175, 215], [120, 170, 235], [90, 150, 220]
rows, titles = [], []
for cid in cases:
    origin = np.load(AVG / f"{cid}_origin.npy")
    gt = trimesh.load(D / cid / "outer_crown.ply", process=False)
    gt = trimesh.Trimesh(np.asarray(gt.vertices) @ R_CF.T - origin, gt.faces, process=False)
    meshes = [(gt, GT), (trimesh.load(AVG / f"{cid}_D.ply", process=False), DC),
              (trimesh.load(ONE / f"{cid}_diff.ply", process=False), D1), (trimesh.load(AVG / f"{cid}_diff.ply", process=False), D8)]
    c = np.asarray(gt.vertices).mean(0)
    cam = F.camera(np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), 35)
    rows.append([F.render([F.sample(m, np.tile(col, (len(m.vertices), 1)), per_mm2=600)], cam, c, 6.5, size=360) for m, col in meshes])
    titles.append(f"{rows_avg[cid]['diff_cd']:.2f} mm")
F.grid(rows, ["技师牙冠", "形状模型", "加细节（采样 1 次）", "加细节（8 次平均）"], titles, out)
print("ok", out)
