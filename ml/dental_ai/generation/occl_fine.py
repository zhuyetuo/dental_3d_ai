"""细小窝沟模型（两步走的第二步）：大、中窝沟用线上的结果（泊松 + 加深），这个模型只在上面补 0.25 mm 以下的细小窝沟。
高度图 0.055 mm 一格（和 occlusal 一样），生成式（flow matching）：沟的位置跟着大窝沟走，细节每次采样可以不同。

- 条件：第一步高度的低频部分（σ 0.25 mm 低通，减顶部 / 2）、它的沟深图、有效、对颌间隙、对颌有效；
- 目标：技师牙冠高度的细尺度部分 = 高度 − σ 0.25 mm 低通。
训练时第一步高度一半用技师自己的低频（教大沟和细沟的关系），一半用粗表面加深 2 倍（和推理时一样），推理时用线上网格自己的高度图。

    python -m dental_ai.generation.occl_fine train --cache ../data/occl_cache/train --val ../data/occl_cache/val --out ../runs/occl/fine.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import occlusal as oc

FINE_SIGMA = 0.25  # mm
SCALE = 20.0  # 细沟 ~0.05 mm × 20 ≈ 1
IN_CH = 5 + 2  # 条件 5 + x_t + t


NOISE_SIGMA = 0.055  # mm：比这还细的是高度图采样噪声 / 网格面片，不算窝沟


def fine_band(h, m):
    """细小窝沟这一段：σ 0.055–0.25 mm。"""
    return oc.lowpass(h, m, NOISE_SIGMA) - oc.lowpass(h, m, FINE_SIGMA)


def cond_from_height(h1, m, gap, ma):
    """第一步高度（减过顶部）→ 5 通道条件。"""
    low = oc.lowpass(h1, m, FINE_SIGMA)
    g = oc.groove_depth_map(low, m)
    return np.stack([low / 2.0, g * 5.0, m.astype(np.float32), gap / 2.0, ma.astype(np.float32)]).astype(np.float32)


def stage1_proxy(hc, mc):
    """推理时的第一步：粗表面加深 2 倍（线上 deepen_grooves(2) 在高度图上的近似）。"""
    return hc - 2.0 * oc.groove_depth_map(hc, mc)


class FineDataset:
    def __init__(self, files, train=False):
        self.files, self.train = files, train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import torch

        z = np.load(self.files[i])
        x, y, m = z["x"].astype(np.float32), z["y"].astype(np.float32), z["m"]
        mc = x[1] > 0
        hc = x[0]
        ht = np.where(m, hc + y, hc)
        mm = m & mc
        tgt = np.where(mm, fine_band(ht, mm), 0.0)
        rng = np.random.default_rng()
        if self.train and rng.random() < 0.5:
            h1 = oc.lowpass(ht, mm, FINE_SIGMA) + rng.normal(0, 0.02)  # 技师自己的低频
        else:
            h1 = stage1_proxy(hc, mc)
        c = cond_from_height(h1, mm, x[2], x[3] > 0)
        if self.train:
            dx, dz = rng.integers(-6, 7, 2)
            c = np.roll(c, (dx, dz), (1, 2))
            tgt = np.roll(tgt, (dx, dz), (0, 1))
            mm = np.roll(mm, (dx, dz), (0, 1))
        return torch.tensor(c), torch.tensor(tgt.astype(np.float32)), torch.tensor(mm)


def sample(model, cond, steps: int = 40, seed: int = 0):
    import torch

    g = torch.Generator(device=cond.device).manual_seed(seed)
    x = torch.randn(cond.shape[0], *cond.shape[2:], device=cond.device, generator=g)
    for k in range(steps):
        t = torch.full_like(x, k / steps)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(torch.cat([cond, x[:, None], t[:, None]], 1)).float()
        x = x + v / steps
    return x / SCALE


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def train(cache, val, out, epochs, batch, lr, workers, max_val):
    import torch
    from torch.utils.data import DataLoader

    files = sorted(cache.glob("*.npz"))
    vfiles = sorted(val.glob("*.npz"))[:max_val]
    dev = "cuda"
    model = oc.build_model(in_ch=IN_CH).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(FineDataset(files, True), batch_size=batch, shuffle=True, num_workers=workers, drop_last=True, persistent_workers=True)
    vdl = DataLoader(FineDataset(vfiles), batch_size=batch, num_workers=workers)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(dl), pct_start=0.05)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, best = time.time(), -1.0
    for ep in range(epochs):
        model.train()
        tl = []
        for c, y, m in dl:
            c, y, m = c.to(dev), y.to(dev), m.to(dev)
            x1 = y * SCALE
            x0 = torch.randn_like(x1)
            t = torch.rand(len(x1), device=dev)[:, None, None]
            xt = (1 - t) * x0 + t * x1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(torch.cat([c, xt[:, None], t.expand_as(xt)[:, None]], 1)).float()
            mf = m.float()
            loss = (((v - (x1 - x0)) ** 2) * mf).sum() / mf.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl.append(loss.item())
        model.eval()
        amp_p, amp_t, cor = [], [], []
        with torch.no_grad():
            for c, y, m in vdl:
                c = c.to(dev)
                d = sample(model, c, steps=20).cpu().numpy()
                y, m = y.numpy(), m.numpy()
                for b in range(len(d)):
                    mb = m[b]
                    if mb.sum() < 500:
                        continue
                    amp_p.append(d[b][mb].std())
                    amp_t.append(y[b][mb].std())
                    cor.append(_corr(d[b][mb], y[b][mb]))
        rec = {"epoch": ep + 1, "train_mse": round(float(np.mean(tl)), 4), "fine_amp_pred": round(float(np.mean(amp_p)), 4),
               "fine_amp_tech": round(float(np.mean(amp_t)), 4), "fine_corr": round(float(np.mean(cor)), 4), "min": round((time.time() - t0) / 60, 1)}
        print(json.dumps(rec), flush=True)
        state = {"state": model.state_dict(), "epoch": ep + 1, "fine": True, "sigma": FINE_SIGMA, "scale": SCALE}
        torch.save(state, out.with_suffix(".last.pt"))
        score = rec["fine_corr"] - abs(rec["fine_amp_pred"] - rec["fine_amp_tech"]) * 5
        if score > best:
            best = score
            torch.save(state, out)


def load(path, device="cuda"):
    import torch

    ck = torch.load(path, map_location=device, weights_only=False)
    m = oc.build_model(in_ch=IN_CH).to(device)
    m.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    return m.eval()


def add_fine(mesh, ant_pts, model, seed: int = 0, gain: float = 1.0, steps: int = 40, denoise: float = 0.0):
    """第一步的网格（线上：泊松 + 加深）→ 细分一次 → 按它自己的咬合面高度图生成细小窝沟、叠上去。返回新网格。"""
    import torch
    import trimesh
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from trimesh.grouping import group_rows

    from ..eval_crownfusion import surface_points

    pts = surface_points(mesh, 300000, 1)
    x4, frame = oc.inputs(pts, ant_pts)
    cx, cz, top = frame
    mc = x4[1] > 0
    h1 = x4[0]
    c = cond_from_height(h1, mc, x4[2], x4[3] > 0)
    with torch.no_grad():
        d = sample(model, torch.tensor(c)[None].cuda(), steps=steps, seed=seed)[0].cpu().numpy()
    if denoise > 0:
        d = ndimage.gaussian_filter(d, denoise)
    resid = (gain * d - fine_band(h1, mc)) * mc  # 换掉第一步自己这一段（多是噪声）
    v, f = trimesh.remesh.subdivide(np.asarray(mesh.vertices, float), np.asarray(mesh.faces))
    m = trimesh.Trimesh(v, f, process=False)
    v = v.copy()
    n = np.asarray(m.vertex_normals)
    inner = ndimage.distance_transform_edt(mc) * oc.CELL
    r = resid * np.clip((inner - 0.4) / 0.8, 0, 1)  # 离咬合面轮廓 0.4 mm 内不动、1.2 mm 以上全量：边缘嵴 / 轴面上不长坑点
    fx = (v[:, 0] - cx) / oc.CELL + oc.RES / 2 - 0.5
    fz = (v[:, 2] - cz) / oc.CELL + oc.RES / 2 - 0.5
    dv = ndimage.map_coordinates(r, [fx, fz], order=1, mode="constant", cval=0.0)
    hv = ndimage.map_coordinates(h1 + top, [fx, fz], order=1, mode="nearest")
    w = np.clip((n[:, 1] - 0.4) / 0.3, 0, 1) * np.clip(1 - (np.abs(v[:, 1] - hv) - 0.1) / 0.2, 0, 1)
    es = m.edges_sorted
    rim = np.unique(es[group_rows(es, require_count=1)])
    if len(rim):
        w *= np.clip(cKDTree(v[rim]).query(v)[0] / 1.0, 0, 1)
    disp = dv * w
    e = m.edges_unique
    deg = np.bincount(e.ravel(), minlength=len(v)).astype(float)
    acc = np.bincount(e[:, 0], disp[e[:, 1]], len(v)) + np.bincount(e[:, 1], disp[e[:, 0]], len(v))
    disp = 0.5 * disp + 0.5 * acc / np.maximum(deg, 1)
    v[:, 1] += disp
    return trimesh.Trimesh(v, f, process=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--val", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=25)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--workers", type=int, default=12)
    t.add_argument("--max-val", type=int, default=200)
    a = ap.parse_args()
    train(a.cache, a.val, a.out, a.epochs, a.batch_size, a.lr, a.workers, a.max_val)


if __name__ == "__main__":
    main()
