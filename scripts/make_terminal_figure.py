"""末端牙（后面没有牙的第二磨牙等）对比：真牙 / 原模型 / 末端牙模型，都用新的外推原点、设计风格 w=4、抹平 0.8。输出 /tmp/terminal.png。"""
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

dev = "cuda"
rows = []
for f in sorted(glob.glob(str(R / "data/radial/bite_val_term/*.npz"))):
    z = np.load(f)
    for k in range(len(z["fdi"])):
        if int(z["fdi"][k]) % 10 == 7:
            rows.append((z["tooth"][k], z["ctx"][k], z["ant"][k], z["has_ant"][k], int(z["fdi"][k])))
pick = np.random.default_rng(5).choice(len(rows), 6, replace=False)
T = lambda i: torch.tensor(np.stack([rows[j][i] for j in pick]).astype(np.float32), device=dev)  # noqa: E731
gt, ctx, ant, has = T(0), T(1), T(2), T(3)
fd = [rows[j][4] for j in pick]
cls = torch.tensor([target_class(f) for f in fd], device=dev)
cond = rd.build_condition(ctx, ant, has)
maps = [gt]
for path in (R / "runs/radial_library_hf/model.pt", R / "runs/radial_term/model.pt"):
    model = rd.load_model(path, dev)
    g = torch.Generator(device=dev).manual_seed(0)
    with torch.no_grad():
        p = rd.decode_tooth(rd.sample(model, cond, cls + rd.DESIGN_OFFSET, 50, g, cls_ref=cls, style_w=4.0))
    maps.append(torch.where(p > 0, rd.lowpass(p, p > 0, 0.8), p))
    m = (p > 0) & (gt > 0)
    print(path.parent.name, "depth err mm", [round(float(((p[i] - gt[i]).abs() * m[i]).sum() / m[i].sum().clamp(min=1)), 2) for i in range(6)])
up = np.array([0.0, 0.0, 1.0])
grid = []
for mp in maps:
    row = []
    for c in range(6):
        d = mp[c].cpu().numpy()
        if (d > 0).sum() < 300:
            row.append(np.ones((300, 300, 3)))
            continue
        msh = map_to_mesh(d, np.zeros(3))
        msh.apply_translation(-msh.vertices.mean(0))
        row.append(F.render([F.sample(msh, np.tile(F.IVORY, (len(msh.vertices), 1)), per_mm2=1400)], F.camera(up, np.array([1.0, 0, 0]), 70), np.zeros(3), 7.0))
    grid.append(row)
F.grid(grid, [str(f) for f in fd], ["natural (scan)", "original model", "terminal-tooth model"], Path("/tmp/terminal.png"), panel=2.2)
