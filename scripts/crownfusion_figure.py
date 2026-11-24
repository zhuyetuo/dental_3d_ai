"""真实预备体（FDI 16 Crown Dataset 测试集）：技师牙冠 / 在真实数据上训过的形状模型 / 原来的形状模型。颊侧看 + 咬合面看。"""
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
from dental_ai.eval_crownfusion import margin_loop  # noqa: E402

for f in font_manager.findSystemFonts():
    if any(k in f for k in ("NotoSansCJK", "NotoSansSC", "wqy", "SourceHanSans", "DroidSansFallback", "simhei")):
        font_manager.fontManager.addfont(f)
        matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break

D = Path.home() / "zyt/dental_data/crownfusion/CrownFusion/test"
NEW, OLD = Path(sys.argv[1]), Path(sys.argv[2])
out = Path(sys.argv[3])
rows_new = {r["case"]: r for r in json.load(open(NEW / "rows.json"))}
cases = sorted(p.name.split("_")[0] for p in NEW.glob("*_gen.ply"))
# 选误差接近中位数的 4 例（不挑最好的）
med = np.median([r["cd"] for r in rows_new.values()])
cases = sorted(cases, key=lambda c: abs(rows_new[c]["cd"] - med))[:4]
PREP, GT, GEN, OLDC, MAR = [200, 200, 205], [235, 205, 120], [120, 170, 235], [190, 150, 220], [220, 40, 40]
rows, titles = [], []
for cid in cases:
    prep = trimesh.load(D / cid / "prep_mesh.ply", process=False)
    crowns = [(trimesh.load(D / cid / "outer_crown.ply", process=False), GT),
              (trimesh.load(NEW / f"{cid}_gen.ply", process=False), GEN),
              (trimesh.load(OLD / f"{cid}_gen.ply", process=False), OLDC)]
    m = margin_loop(prep)
    c = prep.vertices.mean(0) + [0, 2, 0]
    row = []
    for up, toward, el, which in (([0, 1.0, 0], [0, 0, 1.0], 10, (0, 1, 2)), ([0, 0, 1.0], [0, -1.0, 0], 0, (0, 1))):
        cam = F.camera(np.array(up), np.array(toward), el)
        for k in which:
            crown, col = crowns[k]
            parts = [F.sample(crown, np.tile(col, (len(crown.vertices), 1)), per_mm2=120)]
            if toward[2]:  # 颊侧看时把预备体和颈缘线也画上
                parts = [F.sample(prep, np.tile(PREP, (len(prep.vertices), 1)))] + parts + [F.line_part(m, MAR, cam[2])]
            row.append(F.render(parts, cam, c, 8.5, size=360))
    rows.append(row)
    titles.append(f"{rows_new[cid]['cd']:.2f} mm")
F.grid(rows, ["技师牙冠（颊侧）", "真实数据训过（颊侧）", "原来的模型（颊侧）", "技师牙冠（咬合面）", "真实数据训过（咬合面）"], titles, out)
print("ok", out)
