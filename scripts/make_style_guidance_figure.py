"""风格引导 w 的效果图：6 颗后牙，行 = 真牙 / 形状模型 / 网页默认（w=1）/ w=2 / w=3 / w=4。输出 /tmp/style_w.png。用法：fig_w.py [模型] [输出]"""
import glob
import sys
from pathlib import Path

import numpy as np
import torch

R = Path.home() / "zyt/dental_3d_ai"
sys.path.insert(0, str(R / "ml"))
sys.path.insert(0, str(R / "scripts"))
import make_result_figures as F  # noqa: E402
from dental_ai.generation import radial_diffusion as rd  # noqa: E402
from dental_ai.generation.learned import target_class  # noqa: E402
from dental_ai.generation.radial import map_to_mesh  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "models/generation/radial_diffusion.pt"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/style_w.png")
rows = []
for f in sorted(glob.glob(str(R / "data/radial/bite_val/*.npz"))):
    gd = R / "data/radial_d/bite_val" / Path(f).name
    if not gd.exists():
        continue
    z, zd = np.load(f), np.load(gd)
    for k in np.flatnonzero(zd["has_d"].astype(bool)):
        if int(z["fdi"][k]) % 10 in (4, 5, 6, 7):
            rows.append((z["tooth"][k], z["ctx"][k], z["ant"][k], z["has_ant"][k], zd["d"][k], int(z["fdi"][k])))
pick = np.random.default_rng(3).choice(len(rows), 6, replace=False)
T = lambda i: torch.tensor(np.stack([rows[j][i] for j in pick]).astype(np.float32), device=dev)  # noqa: E731
gt, ctx, ant, has, dm = T(0), T(1), T(2), T(3), T(4)
fd = [rows[j][5] for j in pick]
cls = torch.tensor([target_class(f) for f in fd], device=dev)
cond = rd.build_condition(ctx, ant, has)
model = rd.load_model(model_path, dev)
maps = [gt, dm]
for w, sm in ((1.0, 0), (3.0, 0), (4.0, 0.8)):
    g = torch.Generator(device=dev).manual_seed(0)
    with torch.no_grad():
        maps.append(rd.decode_tooth(rd.sample(model, cond, cls + rd.DESIGN_OFFSET, 50, g, init=rd.encode_tooth(dm), strength=0.6, keep_sigma=6,
                                              cls_ref=cls, style_w=w)))
    if sm:
        p = maps[-1]
        maps[-1] = torch.where(p > 0, rd.lowpass(p, p > 0, sm), p)
up = np.array([0.0, 0.0, 1.0])
grid = []
for mp in maps:
    row = []
    for c in range(6):
        m = map_to_mesh(mp[c].cpu().numpy(), np.zeros(3))
        m.apply_translation(-m.vertices.mean(0))
        row.append(F.render([F.sample(m, np.tile(F.IVORY, (len(m.vertices), 1)), per_mm2=1400)], F.camera(up, np.array([1.0, 0, 0]), 75), np.zeros(3), 6.5))
    grid.append(row)
F.grid(grid, [str(f) for f in fd], ["natural (scan)", "shape model", "w=1", "w=3", "w=4 + smooth 0.8"], out_png, panel=2.1)
