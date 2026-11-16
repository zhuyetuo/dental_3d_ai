"""引导区间扫参：w × tmin。分开量“沟”（1–4 像素的起伏，带通）和“麻点”（1 像素以下的起伏，高通）。用法：sweep_tmin.py [模型]"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

R = Path.home() / "zyt/dental_3d_ai"
sys.path.insert(0, str(R / "ml"))
from dental_ai.generation import radial_diffusion as rd  # noqa: E402
from dental_ai.generation.learned import target_class  # noqa: E402

dev = "cuda"
model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else R / "models/generation/radial_diffusion.pt"
rows = []
for f in sorted(glob.glob(str(R / "data/radial/bite_val/*.npz"))):
    gd = R / "data/radial_d/bite_val" / Path(f).name
    if not gd.exists():
        continue
    z, zd = np.load(f), np.load(gd)
    for k in np.flatnonzero(zd["has_d"].astype(bool)):
        rows.append((z["tooth"][k], z["ctx"][k], z["ant"][k], z["has_ant"][k], zd["d"][k], int(z["fdi"][k])))
pick = np.random.default_rng(0).choice(len(rows), 256, replace=False)
T = lambda i: torch.tensor(np.stack([rows[j][i] for j in pick]).astype(np.float32), device=dev)  # noqa: E731
gt, ctx, ant, has, dm = T(0), T(1), T(2), T(3), T(4)
cls = torch.tensor([target_class(rows[j][5]) for j in pick], device=dev)
cond = rd.build_condition(ctx, ant, has)
init = rd.encode_tooth(dm)
model = rd.load_model(model_path, dev)


def band(d, lo, hi):
    m = d > 0
    a = rd.lowpass(d, m, lo) if lo > 0 else d
    b = rd.lowpass(d, m, hi)
    x = torch.where(m, a - b, torch.zeros_like(d))
    return (x.pow(2).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)).sqrt()


def mae(a, b):
    m = (a > 0) & (b > 0)
    return ((a - b).abs() * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)


def say(name, p):
    print(json.dumps({"name": name, "groove": round(float(band(p, 1.0, 4.0).median()), 4), "grain": round(float(band(p, 0, 1.0).median()), 4),
                      "vs_gt": round(float(mae(p, gt).median()), 4)}), flush=True)


say("gt", gt)
say("shape_model", dm)
for w in (3, 4, 5):
    out = []
    for i in range(0, len(pick), 16):
        g = torch.Generator(device=dev).manual_seed(i)
        c = cls[i:i + 16]
        with torch.no_grad():
            out.append(rd.decode_tooth(rd.sample(model, cond[i:i + 16], c + rd.DESIGN_OFFSET, 50, g, init=init[i:i + 16], strength=0.6, keep_sigma=6,
                                                 cls_ref=c, style_w=float(w))))
    p = torch.cat(out)
    say(f"w{w}", p)
    for sm in (0.6, 0.8, 1.0):
        m = p > 0
        say(f"w{w}_smooth{sm}", torch.where(m, rd.lowpass(p, m, sm), p))