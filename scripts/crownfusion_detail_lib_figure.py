"""真实预备体测试集咬合面（斜看 + 正上方）：技师牙冠 / 形状模型 / 加扩散细节 / 加技师牙冠细节库。"""
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
from dental_ai.generation.radial import map_to_mesh  # noqa: E402
from dental_ai.generation.radial_crownfusion import R_CF  # noqa: E402

_fonts = font_manager.findSystemFonts()
for key in ("NotoSansCJK", "NotoSansSC", "SourceHanSans", "wqy", "simhei", "DroidSansFallback"):  # Droid 没有数字和字母，放最后
    hit = [f for f in _fonts if key in f]
    if hit:
        font_manager.fontManager.addfont(hit[0])
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=hit[0]).get_name()
        break

D = Path.home() / "zyt/dental_data/crownfusion/CrownFusion/test"
DIFF, LIB = Path(sys.argv[1]), Path(sys.argv[2])
out = Path(sys.argv[3])
rows_lib = {r["case"]: r for r in json.load(open(LIB / "rows.json"))}
cases = sorted({p.name.split("_")[0] for p in LIB.glob("*_lib.ply")} & {p.name.split("_")[0] for p in DIFF.glob("*_diff.ply")})
med = np.median([r["lib_cd"] for r in rows_lib.values()])
cases = sorted(cases, key=lambda c: abs(rows_lib[c]["lib_cd"] - med))[:4]
cols = [[235, 205, 120], [150, 175, 215], [120, 170, 235], [110, 190, 150]]
rows, titles = [], []
for cid in cases:
    origin = np.load(LIB / f"{cid}_origin.npy")
    gt = trimesh.load(D / cid / "outer_crown.ply", process=False)
    gt = trimesh.Trimesh(np.asarray(gt.vertices) @ R_CF.T - origin, gt.faces, process=False)
    meshes = [gt, trimesh.load(LIB / f"{cid}_D.ply", process=False), trimesh.load(DIFF / f"{cid}_diff.ply", process=False),
              trimesh.load(LIB / f"{cid}_lib.ply", process=False)]
    c = np.asarray(gt.vertices).mean(0)
    row = []
    for up, toward, el in (([0, 0, 1.0], [0, 1.0, 0], 35), ([0, 1.0, 0], [0, 0, -1.0], 0)):
        cam = F.camera(np.array(up), np.array(toward), el)
        row += [F.render([F.sample(m, np.tile(col, (len(m.vertices), 1)), per_mm2=600)], cam, c, 6.5, size=300) for m, col in zip(meshes, cols)]
    rows.append(row)
    titles.append(f"{rows_lib[cid]['lib_cd']:.2f} mm")
names = ["技师牙冠", "形状模型", "加扩散细节", "加细节库"]
F.grid(rows, [n + "（斜看）" for n in names] + names, titles, out)
print("ok", out)
