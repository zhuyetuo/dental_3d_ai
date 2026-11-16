"""同一颗牙、同一个噪声：天然风格 vs 设计风格（牙库），和真牙、对齐的库牙比。
细节量 = 牙冠范围内（深度 − 低通 σ=3 像素）的标准差（mm），越大沟越深越清楚。输出 /tmp/style_cmp.png 和数字。"""
import sys
from pathlib import Path

import numpy as np
import torch

R = Path.home() / "zyt/dental_3d_ai"
sys.path.insert(0, str(R / "ml"))
sys.path.insert(0, str(R / "scripts"))
from dental_ai.generation import radial_diffusion as rd  # noqa: E402
from dental_ai.generation.radial import map_to_mesh  # noqa: E402

dev = "cuda"
val = rd.load_dir([R / "data/radial/bite_val"])
lib = rd.load_lib_dir([R / "data/radial/bite_val"])
# 只用有库牙的那部分（load_lib_dir 的顺序和 load_dir 一致，用 fdi 文件重新配对）
import glob  # noqa: E402

rows = []
for f in sorted(glob.glob(str(R / "data/radial_lib/bite_val/*.npz"))):
    zl, z = np.load(f), np.load(R / "data/radial/bite_val" / Path(f).name)
    for k in np.flatnonzero(zl["ok"] & (zl["score"] <= rd.LIB_MAX_SCORE)):
        rows.append((z["tooth"][k], zl["lib"][k], z["ctx"][k], z["ant"][k], z["has_ant"][k], int(z["fdi"][k])))
rng = np.random.default_rng(0)
pick = rng.choice(len(rows), 128, replace=False)
from dental_ai.generation.learned import target_class  # noqa: E402

T = lambda i: torch.tensor(np.stack([rows[j][i] for j in pick]).astype(np.float32), device=dev)  # noqa: E731
gt, lb, ctx, ant, has = T(0), T(1), T(2), T(3), T(4)
cls = torch.tensor([target_class(rows[j][5]) for j in pick], device=dev)
cond = rd.build_condition(ctx, ant, has)


def detail(d):
    m = d > 0
    low = rd.lowpass(d, m, 3.0)
    hf = torch.where(m, d - low, torch.zeros_like(d))
    return (hf.pow(2).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)).sqrt()


out = {}
for name, path in (("web", R / "models/generation/radial_diffusion.pt"), ("lib", R / "runs/radial_library/model.pt")):
    model = rd.load_model(path, dev)
    styles = [("natural", 0)] + ([("design", rd.DESIGN_OFFSET)] if getattr(model, "design", False) else [])
    for st, off in styles:
        preds = []
        for i in range(0, 128, 32):
            g = torch.Generator(device=dev).manual_seed(i)
            with torch.no_grad():
                preds.append(rd.decode_tooth(rd.sample(model, cond[i:i + 32], cls[i:i + 32] + off, 50, g)))
        out[f"{name}_{st}"] = torch.cat(preds)
print("细节量（mm，中位数）：真牙 %.4f  库牙 %.4f" % (detail(gt).median(), detail(lb).median()))
for k, v in out.items():
    mae_gt = ((v - gt).abs() * ((v > 0) & (gt > 0))).sum((1, 2)) / ((v > 0) & (gt > 0)).sum((1, 2)).clamp(min=1)
    mae_lb = ((v - lb).abs() * ((v > 0) & (lb > 0))).sum((1, 2)) / ((v > 0) & (lb > 0)).sum((1, 2)).clamp(min=1)
    print(f"{k:14s} 细节量 {detail(v).median():.4f}  和真牙差 {mae_gt.median():.3f} mm  和库牙差 {mae_lb.median():.3f} mm")
d = (out["lib_design"] - out["lib_natural"]).abs()
print("同一模型两种风格逐像素差（mm，中位数）", float((d.sum((1, 2)) / ((d > 0).sum((1, 2)).clamp(min=1))).median()))

import make_result_figures as F  # noqa: E402

cols = [int(j) for j in np.argsort(-detail(lb).cpu().numpy())[:6:1]]
grid, up = [], np.array([0.0, 0.0, 1.0])
for maps in (gt, lb, out["lib_natural"], out["lib_design"]):
    row = []
    for c in cols:
        m = map_to_mesh(maps[c].cpu().numpy(), np.zeros(3))
        m.apply_translation(-m.vertices.mean(0))
        row.append(F.render([F.sample(m, np.tile(F.IVORY, (len(m.vertices), 1)), per_mm2=1400)], F.camera(up, np.array([1.0, 0, 0]), 70), np.zeros(3), 6.5))
    grid.append(row)
F.grid(grid, [str(rows[pick[c]][5]) for c in cols], ["natural (scan)", "library aligned", "model: natural", "model: design"], Path("/tmp/style_cmp.png"), panel=2.2)
