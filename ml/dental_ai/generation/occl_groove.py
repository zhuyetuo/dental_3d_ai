"""窝沟图模型：不学整个细节高度，只学“沟在哪、多深”——直接回归技师牙冠的沟深图（闭运算 − 高度，≥ 0）。
用法：粗表面先“填平”自己的沟（闭运算），再按预测的沟深图刻下去：h = closing(粗) − 预测沟深。
损失：L1 + (1 − 相关系数)，相关系数直接对应评估里的“沟深相关”。

    python -m dental_ai.generation.occl_groove train --cache ../data/occl_cache/train --val ../data/occl_cache/val --out ../runs/occl/groove.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import occlusal as oc

IN_CH = 6  # 粗高度、粗有效、对颌间隙、对颌有效、粗细节、粗沟深


def features(x4):
    """缓存里的 4 通道 → 6 通道（numpy）。"""
    mc = x4[1] > 0
    det_c = x4[0] - oc.lowpass(x4[0], mc)
    gc = oc.groove_depth_map(x4[0], mc)
    return np.concatenate([x4, det_c[None], gc[None]]).astype(np.float32)


def scale(x):
    x = x.clone()
    x[:, 0] = x[:, 0] / 2.0
    x[:, 2] = x[:, 2] / 2.0
    x[:, 4] = x[:, 4] * 5.0
    x[:, 5] = x[:, 5] * 5.0
    return x


class GrooveDataset:
    def __init__(self, files, augment=False):
        self.files, self.augment = files, augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import torch

        z = np.load(self.files[i])
        x, y, m = z["x"].astype(np.float32), z["y"].astype(np.float32), z["m"]
        ht = x[0] + y
        tg = oc.groove_depth_map(ht, m).astype(np.float32)
        xf = features(x)
        if self.augment:
            rng = np.random.default_rng()
            dx, dz = rng.integers(-6, 7, 2)
            xf = np.roll(xf, (dx, dz), (1, 2))
            tg = np.roll(tg, (dx, dz), (0, 1))
            m = np.roll(m, (dx, dz), (0, 1))
        return torch.tensor(xf), torch.tensor(tg), torch.tensor(m)


def _corr(p, t, mf):
    n = mf.sum((1, 2)).clamp(min=1)
    pm = (p * mf).sum((1, 2)) / n
    tm = (t * mf).sum((1, 2)) / n
    pc, tc = (p - pm[:, None, None]) * mf, (t - tm[:, None, None]) * mf
    return (pc * tc).sum((1, 2)) / (pc.pow(2).sum((1, 2)).sqrt() * tc.pow(2).sum((1, 2)).sqrt() + 1e-8)


def train(cache, val, out, epochs, batch, lr, workers, max_val):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    files = sorted(cache.glob("*.npz"))
    vfiles = sorted(val.glob("*.npz"))[:max_val]
    dev = "cuda"
    model = oc.build_model(in_ch=IN_CH).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(GrooveDataset(files, True), batch_size=batch, shuffle=True, num_workers=workers, drop_last=True, persistent_workers=True)
    vdl = DataLoader(GrooveDataset(vfiles), batch_size=batch, num_workers=workers)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(dl), pct_start=0.05)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, best = time.time(), -1
    for ep in range(epochs):
        model.train()
        tl = []
        for x, t, m in dl:
            x, t, m = x.to(dev), t.to(dev), m.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = F.softplus(model(scale(x)).float()) / 5.0
            mf = m.float()
            l1 = ((p - t).abs() * mf).sum() / mf.sum().clamp(min=1)
            loss = 5.0 * l1 + (1 - _corr(p, t, mf)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl.append(loss.item())
        model.eval()
        cs, c0, ious, iou0 = [], [], [], []
        with torch.no_grad():
            for x, t, m in vdl:
                x, t, m = x.to(dev), t.to(dev), m.to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = F.softplus(model(scale(x)).float()) / 5.0
                mf = m.float()
                cs += _corr(p, t, mf).tolist()
                c0 += _corr(x[:, 5], t, mf).tolist()
                for b in range(len(x)):
                    mb = m[b]
                    a, bb, a0 = p[b][mb] > 0.08, t[b][mb] > 0.08, x[b, 5][mb] > 0.08
                    ious.append(((a & bb).sum() / (a | bb).sum().clamp(min=1)).item())
                    iou0.append(((a0 & bb).sum() / (a0 | bb).sum().clamp(min=1)).item())
        v = float(np.nanmean(cs))
        print(json.dumps({"epoch": ep + 1, "loss": round(float(np.mean(tl)), 4), "val_corr": round(v, 4), "val_corr_coarse": round(float(np.nanmean(c0)), 4),
                          "val_iou": round(float(np.mean(ious)), 4), "val_iou_coarse": round(float(np.mean(iou0)), 4),
                          "min": round((time.time() - t0) / 60, 1)}), flush=True)
        state = {"state": model.state_dict(), "epoch": ep + 1, "val": v, "groove": True, "res": oc.RES, "cell": oc.CELL}
        torch.save(state, out.with_suffix(".last.pt"))
        if v > best:
            best = v
            torch.save(state, out)


def load(path, device="cuda"):
    import torch

    ck = torch.load(path, map_location=device, weights_only=False)
    m = oc.build_model(in_ch=IN_CH).to(device)
    m.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    return m.eval()


def predict_map(model, x4):
    """4 通道 → (预测沟深图, 粗沟深图)，numpy。"""
    import torch
    import torch.nn.functional as F

    xf = features(x4)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        p = F.softplus(model(scale(torch.tensor(xf)[None].cuda())).float()) / 5.0
    return p[0].float().cpu().numpy(), xf[5]


def carve(hc, mc, pred_g, gc, gain: float = 1.0):
    """粗高度填平自己的沟，再刻预测的沟：h = hc + gc − gain × pred。返回要加到 hc 上的残差。"""
    return (gc - gain * pred_g) * mc


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
    t.add_argument("--workers", type=int, default=10)
    t.add_argument("--max-val", type=int, default=300)
    a = ap.parse_args()
    train(a.cache, a.val, a.out, a.epochs, a.batch_size, a.lr, a.workers, a.max_val)


if __name__ == "__main__":
    main()
