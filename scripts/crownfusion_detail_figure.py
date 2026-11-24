"""真实预备体测试集的咬合面：技师牙冠 / 形状模型 / 形状模型 + 扩散模型的细节（咬合面正上方看、斜着看）。"""
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
G = Path(sys.argv[1])
out = Path(sys.argv[2])
rows_all = {r["case"]: r for r in json.load(open(G / "rows.json"))}
cases = sorted({p.name.split("_")[0] for p in G.glob("*_diff.ply")})
med = np.median([r["diff_cd"] for r in rows_all.values()])
cases = sorted(cases, key=lambda c: abs(rows_all[c]["diff_cd"] - med))[:4]
GT, DC, DF = [235, 205, 120], [150, 175, 215], [120, 170, 235]
rows, titles = [], []
for cid in cases:
    origin = np.load(G / f"{cid}_origin.npy")
    gt = trimesh.load(D / cid / "outer_crown.ply", process=False)
    gt = trimesh.Trimesh(np.asarray(gt.vertices) @ R_CF.T - origin, gt.faces, process=False)
    meshes = [(gt, GT), (trimesh.load(G / f"{cid}_D.ply", process=False), DC), (trimesh.load(G / f"{cid}_diff.ply", process=False), DF)]
    c = np.asarray(gt.vertices).mean(0)
    row = []
    for up, toward, el in (([0, 1.0, 0], [0, 0, -1.0], 0), ([0, 0, 1.0], [0, 1.0, 0], 35)):
        cam = F.camera(np.array(up), np.array(toward), el)
        for m, col in meshes:
            row.append(F.render([F.sample(m, np.tile(col, (len(m.vertices), 1)), per_mm2=600)], cam, c, 6.5, size=360))
    rows.append(row)
    r = rows_all[cid]
    titles.append(f"{r['diff_cd']:.2f} mm")
F.grid(rows, ["技师牙冠", "形状模型", "加扩散细节", "技师牙冠（斜看）", "形状模型（斜看）", "加扩散细节（斜看）"], titles, out)
print("ok", out)
