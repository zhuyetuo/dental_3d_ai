"""咬合面单独一个模型（两步走的第二步）：整体形状由体素模型 + 泊松出，咬合面剥成高度图，2D 网络补窝沟牙尖细节，再叠回去。

坐标：数据坐标（y 朝咬合面，x 近中 → 远中，z 颊舌），和 voxel_crown 的 R_DATA 一致；26 号在流程里已镜像成 16。
高度图：以粗咬合面顶部的中心为中心，RES × RES 格、CELL mm 一格（256 × 0.055 = 14 mm，盖住磨牙咬合面），每格取该处表面最高点。
- 输入：粗高度（减去顶部参考高度）、粗高度有效、对颌最低面到粗高度的间隙（对颌牙尖决定窝的位置）、对颌有效；
- 目标：技师牙冠高度 − 粗高度（残差），只在两者都有效的格子上算损失。

    python -m dental_ai.generation.occlusal build --data ~/zyt/dental_data/crownfusion/CrownFusion --split train --out ../data/occl_cache/train
    python -m dental_ai.generation.occlusal train --cache ../data/occl_cache/train --val ../data/occl_cache/val --out ../runs/occl/occl.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

RES, CELL = 256, 0.055
TOP_BAND = 5.0  # 只看最高处往下 5 mm 内的表面（咬合面 + 上部轴面）
DETAIL_SIGMA = 0.6  # mm：比这细的起伏（窝沟、牙尖尖端、边缘嵴）算“细节”，模型只学这部分


def lowpass(h, m, sigma_mm: float = DETAIL_SIGMA):
    """有效区内的高斯低通（归一化卷积，边缘不被 0 拉低）。"""
    from scipy import ndimage

    s = sigma_mm / CELL
    mf = m.astype(float)
    num = ndimage.gaussian_filter(h * mf, s)
    den = ndimage.gaussian_filter(mf, s)
    return np.where(m, num / np.maximum(den, 1e-6), 0.0)


def _grid_max(p, cx, cz, top0, mode="max"):
    """点 → 高度图（每格最高 / 最低），返回 (高度, 有效)。"""
    ix = np.floor((p[:, 0] - cx) / CELL + RES / 2).astype(int)
    iz = np.floor((p[:, 2] - cz) / CELL + RES / 2).astype(int)
    ok = (ix >= 0) & (ix < RES) & (iz >= 0) & (iz < RES)
    ix, iz, h = ix[ok], iz[ok], p[ok, 1]
    if mode == "max":
        g = np.full((RES, RES), -np.inf)
        np.maximum.at(g, (ix, iz), h)
    else:
        g = np.full((RES, RES), np.inf)
        np.minimum.at(g, (ix, iz), h)
    return g, np.isfinite(g)


def _fill(g, have, iters: int = 2):
    """小洞补上（最近有效格的值），有效区域做一次闭运算。"""
    from scipy import ndimage

    if not have.any():
        return np.zeros_like(g), have
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(have, iterations=iters))
    idx = ndimage.distance_transform_edt(~have, return_distances=False, return_indices=True)
    return np.where(mask, g[tuple(idx)], 0.0), mask


def frame_from_coarse(coarse_pts):
    y = coarse_pts[:, 1]
    top = np.percentile(y, 99)
    sel = coarse_pts[y >= top - TOP_BAND]
    return float(sel[:, 0].mean()), float(sel[:, 2].mean()), float(top)


def coarse_map(coarse_pts, cx, cz, top):
    p = coarse_pts[coarse_pts[:, 1] >= top - TOP_BAND]
    g, have = _grid_max(p, cx, cz, top)
    return _fill(g, have)


def antagonist_map(ant_pts, cx, cz, top, hc, mc):
    """对颌（在 +y 方向）每格最低点；间隙 = 对颌 − 粗高度，裁到 [0, 4] mm。对颌点稀，先在 2 倍格上取再放大。"""
    from scipy import ndimage

    if ant_pts is None or len(ant_pts) < 50:
        return np.zeros((RES, RES)), np.zeros((RES, RES), bool)
    a = ant_pts[(ant_pts[:, 1] > top - 2.0) & (ant_pts[:, 1] < top + 10.0)]
    if len(a) < 50:
        return np.zeros((RES, RES)), np.zeros((RES, RES), bool)
    g, have = _grid_max(a, cx, cz, top, mode="min")
    g = np.where(have, g, np.inf)
    g2 = ndimage.minimum_filter(g, size=5)  # 稀疏点：5 格内取最低
    have2 = np.isfinite(g2)
    gf, m = _fill(np.where(have2, g2, 0.0), have2, iters=3)
    gf = ndimage.gaussian_filter(gf, 1.5)
    gap = np.clip(gf - (hc + top), 0, 4.0)
    return gap, m & mc


def inputs(coarse_pts, ant_pts):
    cx, cz, top = frame_from_coarse(coarse_pts)
    hc, mc = coarse_map(coarse_pts, cx, cz, top)
    hc = np.where(mc, hc - top, 0.0)
    gap, ma = antagonist_map(ant_pts, cx, cz, top, hc, mc)
    x = np.stack([hc, mc, gap, ma]).astype(np.float32)
    return x, (cx, cz, top)


def _build_case(args):
    import torch
    import trimesh

    from ..eval_crownfusion import load_case, surface_points
    from . import voxel_crown as vc
    from .poisson import poisson_mesh

    d, out = args
    dest = out / f"{d.name}.npz"
    if dest.exists():
        return True
    global _VOX
    try:
        if "_VOX" not in globals():
            _VOX = vc.load(Path(__import__("os").environ["OCCL_VOX"]))
        pts, props, prep, crown = load_case(d)
        pp = surface_points(prep, vc.N_PREP, int(d.name) + 1)
        c = vc.crop_center(pp)
        bt = {"prep": torch.tensor(pp - c, dtype=torch.float32)[None],
              "ctx": torch.tensor(np.concatenate([pts[:, :3] - c, (props[:, 1:2] + 1)], 1), dtype=torch.float32)[None]}
        p, n = vc.predict_points(_VOX, vc.make_inputs(bt, "cuda"))[0]
        p, n = p.float().cpu().numpy() + c, n.float().cpu().numpy()
        pm = poisson_mesh(p, n, 192, 1.0, 0.3)
        coarse = np.asarray(trimesh.sample.sample_surface(pm, 300000, seed=1)[0])
        ant = pts[props[:, 1] == 1, :3].astype(float)
        x, (cx, cz, top) = inputs(coarse, ant)
        tech = np.asarray(trimesh.sample.sample_surface(crown, 300000, seed=2)[0])
        ht, mt = coarse_map(tech, cx, cz, top)
        tgt = np.where(mt & (x[1] > 0), (ht - top) - x[0], 0.0)
        m = mt & (x[1] > 0) & (np.abs(tgt) < 2.0)
        np.savez_compressed(dest, x=x.astype(np.float16), y=tgt.astype(np.float16), m=m, frame=np.array([cx, cz, top], np.float32))
        return True
    except Exception as exc:  # noqa: BLE001
        print("skip", d.name, type(exc).__name__, str(exc)[:120], flush=True)
        return False


def build(data: Path, split: str, out: Path, workers: int, limit: int | None, vox: Path) -> None:
    import multiprocessing as mp
    import os

    os.environ["OCCL_VOX"] = str(vox)
    ids = json.load(open(data / {"test": "testing.json", "val": "validation.json", "train": "training.json"}[split]))
    ids = sorted(ids, key=int)[:limit]
    out.mkdir(parents=True, exist_ok=True)
    t0, ok = time.time(), 0
    with mp.get_context("spawn").Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_build_case, [(data / split / str(c), out) for c in ids], chunksize=4)):
            ok += bool(r)
            if k % 200 == 0:
                print(json.dumps({"done": k, "ok": ok, "min": round((time.time() - t0) / 60, 1)}), flush=True)
    print(json.dumps({"split": split, "ok": ok, "n": len(ids), "min": round((time.time() - t0) / 60, 1)}), flush=True)


# ---------------- 模型 ----------------

def build_model(f0: int = 32, levels: int = 5, in_ch: int = 5):
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.c = nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(),
                                   nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU())
            self.s = nn.Conv2d(a, b, 1) if a != b else nn.Identity()

        def forward(self, x):
            return self.c(x) + self.s(x)

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            ch = [f0 * 2 ** i for i in range(levels)]
            self.inp = Block(in_ch, ch[0])
            self.down = nn.ModuleList([Block(ch[i], ch[i + 1]) for i in range(levels - 1)])
            self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(levels - 1))])
            self.dec = nn.ModuleList([Block(ch[i] * 2, ch[i]) for i in reversed(range(levels - 1))])
            self.head = nn.Conv2d(ch[0], 1, 1)
            self.pool = nn.AvgPool2d(2)

        def forward(self, x):
            h = self.inp(x)
            skips = [h]
            for d in self.down:
                h = d(self.pool(h))
                skips.append(h)
            for j, (u, d) in enumerate(zip(self.up, self.dec)):
                h = d(torch.cat([u(h), skips[-2 - j]], 1))
            return self.head(h)[:, 0]

    import torch

    return UNet()


def _scale(x):
    """输入归一化：高度 / 2 mm，间隙 / 2 mm。"""
    x = x.clone()
    x[:, 0] = x[:, 0] / 2.0
    x[:, 2] = x[:, 2] / 2.0
    x[:, 4] = x[:, 4] * 5.0  # 细节通道很小，放大到和别的通道同量级
    return x


class OcclDataset:
    def __init__(self, files, augment: bool = False):
        self.files, self.augment = files, augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import torch

        z = np.load(self.files[i])
        x, y, m = z["x"].astype(np.float32), z["y"].astype(np.float32), z["m"]
        mc = x[1] > 0
        ht = x[0] + y  # 技师高度（相对粗表面的顶部参考）
        y = np.where(m, ht - lowpass(np.where(mc, ht, x[0]), mc), 0.0).astype(np.float32)  # 技师细节
        x = np.concatenate([x, (x[0] - lowpass(x[0], mc))[None].astype(np.float32)])  # 加一个通道：粗表面自己的细节
        if self.augment:  # 小平移（中心估计有误差），不翻转（解剖方向固定）
            rng = np.random.default_rng()
            dx, dz = rng.integers(-6, 7, 2)
            x = np.roll(x, (dx, dz), (1, 2))
            y = np.roll(y, (dx, dz), (0, 1))
            m = np.roll(m, (dx, dz), (0, 1))
        return torch.tensor(x), torch.tensor(y), torch.tensor(m)


def _loss(pred, y, m, x):
    import torch
    import torch.nn.functional as F

    mf = m.float()
    l1 = (torch.abs(pred - y) * mf).sum() / mf.sum().clamp(min=1)
    # 梯度损失：沟壁 / 牙尖的陡度要对（只回归高度会被抹平）
    gx = lambda t: t[:, 1:, :] - t[:, :-1, :]  # noqa: E731
    gz = lambda t: t[:, :, 1:] - t[:, :, :-1]  # noqa: E731
    full_p, full_y = pred, y
    mx, mz = mf[:, 1:, :] * mf[:, :-1, :], mf[:, :, 1:] * mf[:, :, :-1]
    lg = ((torch.abs(gx(full_p) - gx(full_y)) * mx).sum() / mx.sum().clamp(min=1)
          + (torch.abs(gz(full_p) - gz(full_y)) * mz).sum() / mz.sum().clamp(min=1)) / CELL
    return l1 + 0.3 * lg, l1


def groove_depth_map(h, valid, size: int = 23):
    """高度图上窄沟的深度（闭运算 − 原高度），size 格 ≈ 1.3 mm。"""
    from scipy import ndimage

    g = ndimage.gaussian_filter(np.where(valid, h, h[valid].min() if valid.any() else 0), 1.0)
    return (ndimage.grey_closing(g, size=(size, size)) - g) * valid


def train(cache: Path, val: Path, out: Path, epochs: int, batch: int, lr: float, workers: int, max_val: int) -> None:
    import torch
    from torch.utils.data import DataLoader

    files = sorted(cache.glob("*.npz"))
    vfiles = sorted(val.glob("*.npz"))[:max_val]
    dev = "cuda"
    model = build_model().to(dev)
    print(json.dumps({"train": len(files), "val": len(vfiles), "params": sum(p.numel() for p in model.parameters())}), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(OcclDataset(files, True), batch_size=batch, shuffle=True, num_workers=workers, drop_last=True, persistent_workers=True)
    vdl = DataLoader(OcclDataset(vfiles), batch_size=batch, num_workers=workers)
    steps = epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    best = 1e9
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tl = []
        for x, y, m in dl:
            x, y, m = x.to(dev), y.to(dev), m.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(_scale(x))
            loss, l1 = _loss(pred.float(), y, m, x)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl.append(l1.item())
        model.eval()
        vl, v0, gd_p, gd_t, gd_0 = [], [], [], [], []
        with torch.no_grad():
            for x, y, m in vdl:
                x, y, m = x.to(dev), y.to(dev), m.to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(_scale(x)).float()
                mf = m.float()
                vl.append(((pred - y).abs() * mf).sum().item() / mf.sum().item())
                v0.append(((x[:, 4] - y).abs() * mf).sum().item() / mf.sum().item())  # 基线：保留粗表面自己的细节
                for b in range(min(4, len(x))):
                    hc, mv = x[b, 0].cpu().numpy(), m[b].cpu().numpy()
                    low = hc - x[b, 4].cpu().numpy()
                    gd_p.append(np.percentile(groove_depth_map(low + pred[b].cpu().numpy(), mv)[mv], 90))
                    gd_t.append(np.percentile(groove_depth_map(low + y[b].cpu().numpy(), mv)[mv], 90))
                    gd_0.append(np.percentile(groove_depth_map(hc, mv)[mv], 90))
        v = float(np.mean(vl))
        rec = {"epoch": ep + 1, "train_l1": round(float(np.mean(tl)), 4), "val_l1": round(v, 4), "val_l1_coarse": round(float(np.mean(v0)), 4),
               "groove_pred": round(float(np.mean(gd_p)), 3), "groove_tech": round(float(np.mean(gd_t)), 3), "groove_coarse": round(float(np.mean(gd_0)), 3),
               "min": round((time.time() - t0) / 60, 1)}
        print(json.dumps(rec), flush=True)
        state = {"state": model.state_dict(), "epoch": ep + 1, "val": v, "res": RES, "cell": CELL}
        torch.save(state, out.with_suffix(".last.pt"))
        if v < best:
            best = v
            torch.save(state, out)


DETAIL_SCALE = 10.0  # 细节（mm）× 10 后再加噪声，量级 ~1


def load(path: Path, device: str = "cuda"):
    import torch

    ck = torch.load(path, map_location=device, weights_only=False)
    m = build_model(in_ch=7 if ck.get("flow") else 5).to(device)
    m.load_state_dict({k: v.float() for k, v in ck["state"].items()})
    m.flow = bool(ck.get("flow"))
    return m.eval()


def flow_sample(model, cond, steps: int = 20, seed: int = 0):
    """cond (B,5,H,W) 已归一化 → 细节 (B,H,W)，mm。"""
    import torch

    g = torch.Generator(device=cond.device).manual_seed(seed)
    x = torch.randn(cond.shape[0], *cond.shape[2:], device=cond.device, generator=g)
    for k in range(steps):
        t = torch.full_like(x, k / steps)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(torch.cat([cond, x[:, None], t[:, None]], 1)).float()
        x = x + v / steps
    return x / DETAIL_SCALE


def train_flow(cache: Path, val: Path, out: Path, epochs: int, batch: int, lr: float, workers: int, max_val: int) -> None:
    import torch
    from torch.utils.data import DataLoader

    files = sorted(cache.glob("*.npz"))
    vfiles = sorted(val.glob("*.npz"))[:max_val]
    dev = "cuda"
    model = build_model(in_ch=7).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = DataLoader(OcclDataset(files, True), batch_size=batch, shuffle=True, num_workers=workers, drop_last=True, persistent_workers=True)
    vdl = DataLoader(OcclDataset(vfiles), batch_size=batch, num_workers=workers)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * len(dl), pct_start=0.05)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, best = time.time(), 1e9
    for ep in range(epochs):
        model.train()
        tl = []
        for x, y, m in dl:
            x, y, m = x.to(dev), y.to(dev), m.to(dev)
            cond = _scale(x)
            x1 = y * DETAIL_SCALE
            x0 = torch.randn_like(x1)
            t = torch.rand(len(x1), device=dev)[:, None, None]
            xt = (1 - t) * x0 + t * x1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(torch.cat([cond, xt[:, None], t.expand_as(xt)[:, None]], 1)).float()
            mf = m.float()
            loss = (((v - (x1 - x0)) ** 2) * mf).sum() / mf.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl.append(loss.item())
        model.eval()
        vl, gd_p, gd_t = [], [], []
        with torch.no_grad():
            for x, y, m in vdl:
                x, y, m = x.to(dev), y.to(dev), m.to(dev)
                d = flow_sample(model, _scale(x))
                mf = m.float()
                vl.append(((d - y).abs() * mf).sum().item() / mf.sum().item())
                for b in range(min(4, len(x))):
                    hc, mv = x[b, 0].cpu().numpy(), m[b].cpu().numpy()
                    low = hc - x[b, 4].cpu().numpy()
                    gd_p.append(np.percentile(groove_depth_map(low + d[b].cpu().numpy(), mv)[mv], 90))
                    gd_t.append(np.percentile(groove_depth_map(low + y[b].cpu().numpy(), mv)[mv], 90))
        v_ = float(np.mean(vl))
        print(json.dumps({"epoch": ep + 1, "train_mse": round(float(np.mean(tl)), 4), "val_l1_sample": round(v_, 4),
                          "groove_pred": round(float(np.mean(gd_p)), 3), "groove_tech": round(float(np.mean(gd_t)), 3),
                          "min": round((time.time() - t0) / 60, 1)}), flush=True)
        state = {"state": model.state_dict(), "epoch": ep + 1, "val": v_, "flow": True, "res": RES, "cell": CELL}
        torch.save(state, out.with_suffix(".last.pt"))
        if v_ < best:
            best = v_
            torch.save(state, out)


def predict(model, coarse_pts, ant_pts):
    """返回 (残差高度图, 有效, frame)。"""
    import torch

    x, frame = inputs(coarse_pts, ant_pts)
    mc = x[1] > 0
    det_c = x[0] - lowpass(x[0], mc)
    x5 = np.concatenate([x, det_c[None]]).astype(np.float32)
    with torch.no_grad():
        cond = _scale(torch.tensor(x5)[None].cuda())
        if getattr(model, "flow", False):
            d = flow_sample(model, cond)[0].cpu().numpy()
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d = model(cond).float()[0].cpu().numpy()
    return (d - det_c) * mc, mc, frame  # 把粗表面的细节换成预测的细节


def apply_residual(m, resid, valid, frame, blend: float = 0.8):
    """把残差高度叠到网格上：朝咬合面、在高度图表面上的顶点沿 y 挪 resid；有效区边缘 blend mm 内渐变到 0，底边不动。"""
    import trimesh
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from trimesh.grouping import group_rows

    cx, cz, top = frame
    v = np.asarray(m.vertices, float).copy()
    n = np.asarray(m.vertex_normals)
    edge = ndimage.distance_transform_edt(valid) * CELL
    r = resid * np.clip(edge / blend, 0, 1)
    pts = np.asarray(trimesh.sample.sample_surface(m, 300000, seed=3)[0])
    hc, mc = coarse_map(pts, cx, cz, top)
    fx = (v[:, 0] - cx) / CELL + RES / 2 - 0.5
    fz = (v[:, 2] - cz) / CELL + RES / 2 - 0.5
    dv = ndimage.map_coordinates(r, [fx, fz], order=1, mode="constant", cval=0.0)
    hv = ndimage.map_coordinates(hc, [fx, fz], order=1, mode="nearest")
    w = np.clip((n[:, 1] - 0.2) / 0.4, 0, 1) * np.clip(1 - (np.abs(v[:, 1] - hv) - 0.2) / 0.3, 0, 1) * (v[:, 1] >= top - TOP_BAND)
    es = m.edges_sorted
    rim = np.unique(es[group_rows(es, require_count=1)])
    if len(rim):
        w *= np.clip(cKDTree(v[rim]).query(v)[0] / 1.0, 0, 1)
    disp = dv * w
    e = m.edges_unique
    deg = np.bincount(e.ravel(), minlength=len(v)).astype(float)
    for _ in range(3):
        acc = np.bincount(e[:, 0], disp[e[:, 1]], len(v)) + np.bincount(e[:, 1], disp[e[:, 0]], len(v))
        disp = 0.5 * disp + 0.5 * acc / np.maximum(deg, 1)
    v[:, 1] += disp
    return trimesh.Trimesh(v, m.faces, process=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", type=Path, required=True)
    b.add_argument("--split", default="train")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--limit", type=int)
    b.add_argument("--vox", type=Path, default=Path("../models/generation/voxel_crown.pt"))
    for name in ("train", "trainflow"):
        t = sub.add_parser(name)
        t.add_argument("--cache", type=Path, required=True)
        t.add_argument("--val", type=Path, required=True)
        t.add_argument("--out", type=Path, required=True)
        t.add_argument("--epochs", type=int, default=30)
        t.add_argument("--batch-size", type=int, default=16)
        t.add_argument("--lr", type=float, default=1e-3)
        t.add_argument("--workers", type=int, default=8)
        t.add_argument("--max-val", type=int, default=300)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.data, a.split, a.out, a.workers, a.limit, a.vox.resolve())
    elif a.cmd == "trainflow":
        train_flow(a.cache, a.val, a.out, a.epochs, a.batch_size, a.lr, a.workers, a.max_val)
    else:
        train(a.cache, a.val, a.out, a.epochs, a.batch_size, a.lr, a.workers, a.max_val)


if __name__ == "__main__":
    main()
