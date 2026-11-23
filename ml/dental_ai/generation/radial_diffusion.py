"""径向深度图上的扩散模型：看着其余牙齿和对颌的深度图（逐像素对齐），生成缺失牙冠的深度图。

    python -m dental_ai.generation.radial_diffusion train --data ../data/radial/train ../data/radial/bite --val ../data/radial/bite_val \\
        --out ../runs/radial_diffusion/model.pt
    python -m dental_ai.generation.radial_diffusion eval --model ../runs/radial_diffusion/model.pt --val ../data/radial/bite_val

表示见 radial.py。网络输入：加噪的牙冠图（深度 + 有没有牙冠，2 通道）、条件图 5 通道（其余牙齿的深度和有无、对颌的深度和有无、
投影圆盘），加上扩散步数和牙位。输出按 v-prediction 训练，采样用 DDIM。
"""

from __future__ import annotations

import argparse
import json
import os
import math
import time
from pathlib import Path

import numpy as np

from .radial import SIZE, pixel_directions

TOOTH_SCALE = 5.0  # 牙冠深度：r / 5 − 1（牙冠半径 1.5–9 mm → −0.7–0.8）；没有牙冠的像素 −1
FAR_SCALE = 8.0  # 上下文 / 对颌：r / 8 − 1（最远 16 mm → 1）；没有东西的像素 1（“很远”）
STEPS = 1000


def d_dir_for(data_dir: Path) -> Path:
    """data/radial/<split> 对应的 D 深度图目录 data/radial_d/<split>（radial_dmaps.py 生成）。"""
    data_dir = Path(data_dir)
    return data_dir.parent.parent / "radial_d" / data_dir.name


def load_dir(dirs: list[Path], with_d: bool = False):
    """with_d：同时读 D（学习模型）在每颗牙上的输出深度图，只留 D 生成成功的牙。返回的元组末尾多一个 d。"""
    from .learned import target_class

    tooth, ctx, ant, has, cls, dmap = [], [], [], [], [], []
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            if with_d:
                g = d_dir_for(d) / f.name
                if not g.exists():
                    continue
            with np.load(f) as z:
                keep = np.ones(len(z["fdi"]), bool)
                if with_d:
                    with np.load(g) as zd:
                        assert (zd["fdi"] == z["fdi"]).all(), f
                        keep = zd["has_d"].astype(bool)
                        dmap.append(zd["d"][keep])
                tooth.append(z["tooth"][keep]), ctx.append(z["ctx"][keep]), ant.append(z["ant"][keep]), has.append(z["has_ant"][keep])
                cls.append(np.array([target_class(int(v)) for v in z["fdi"][keep]], np.int64))
    out = (np.concatenate(tooth), np.concatenate(ctx), np.concatenate(ant), np.concatenate(has), np.concatenate(cls))
    return out + (np.concatenate(dmap),) if with_d else out


LIB_MAX_SCORE = float(os.environ.get("LIB_MAX_SCORE", "0.5"))  # mm：库牙和真牙对齐后的双向平均距离上限
DESIGN_OFFSET = 16  # 设计风格（牙库样本）的牙位类别 = 天然牙的类别 + 16


def load_lib_dir(dirs: list[Path]):
    """牙库造的设计风格样本（radial_library.py build 生成，data/radial_lib/<split>）：答案换成对齐过来的库牙，条件不变，类别 + 16。"""
    from .learned import target_class
    from .radial_library import lib_dir_for

    tooth, ctx, ant, has, cls = [], [], [], [], []
    for d in dirs:
        ld = lib_dir_for(Path(d))
        for g in sorted(ld.glob("*.npz")):
            with np.load(g) as zl, np.load(Path(d) / g.name) as z:
                assert (zl["fdi"] == z["fdi"]).all(), g
                keep = zl["ok"].astype(bool) & (zl["score"] <= LIB_MAX_SCORE)  # 对得不好的库牙不要
                if not keep.any():
                    continue
                tooth.append(zl["lib"][keep]), ctx.append(z["ctx"][keep]), ant.append(z["ant"][keep]), has.append(z["has_ant"][keep])
                cls.append(np.array([target_class(int(v)) + DESIGN_OFFSET for v in z["fdi"][keep]], np.int64))
    if not tooth:
        return None
    return np.concatenate(tooth), np.concatenate(ctx), np.concatenate(ant), np.concatenate(has), np.concatenate(cls)


def jitter_d(d, generator=None):
    """训练时给 D 的深度图加扰动：训练牙上 D 见过答案，输出比没见过的牙好，不加扰动模型会学成“照抄 D”。
    整体缩放 ±4%、整体平移 ±0.2 mm、再加一层平滑的起伏（约 ±0.15 mm）。d：(B, S, S) mm，0 = 没有。"""
    import torch
    import torch.nn.functional as F

    b = len(d)
    m = d > 0
    scale = 1 + 0.04 * torch.randn(b, 1, 1, device=d.device, generator=generator)
    shift = 0.2 * torch.randn(b, 1, 1, device=d.device, generator=generator)
    coarse = torch.randn(b, 1, 10, 10, device=d.device, generator=generator) * 0.15
    wave = F.interpolate(coarse, size=d.shape[-2:], mode="bicubic", align_corners=False)[:, 0]
    return torch.where(m, (d * scale + shift + wave).clamp(min=0.31), d)


def encode_tooth(t):  # (..., S, S) mm，0 = 没有 → (..., 2, S, S)
    import torch

    m = t > 0
    return torch.stack([torch.where(m, t / TOOTH_SCALE - 1, -torch.ones_like(t)), torch.where(m, 1.0, -1.0)], dim=-3)


def encode_far(c):
    import torch

    m = c > 0
    return torch.stack([torch.where(m, c / FAR_SCALE - 1, torch.ones_like(c)), torch.where(m, 1.0, -1.0)], dim=-3)


def decode_tooth(x):  # (..., 2, S, S) → 深度 mm（没有牙冠的像素 0）
    import torch

    return torch.where(x[..., 1, :, :] > 0, (x[..., 0, :, :] + 1) * TOOTH_SCALE, torch.zeros_like(x[..., 0, :, :]))


MARGIN_DROP = 0.15  # 训练时不给颈缘线的比例


def margin_edge(tooth):
    """答案的径向图 (B, S, S) mm → 颈缘线图：掩码的外边缘像素（一圈），值是那里的深度，别处 0。投影圆盘最外一圈不算（那是视野边缘）。"""
    import torch
    import torch.nn.functional as F

    m = (tooth > 0).float()
    inner = -F.max_pool2d(-m[:, None], 3, 1, 1)[:, 0]
    c = (torch.arange(SIZE, device=tooth.device) + 0.5) / SIZE * 2 - 1
    rho = torch.hypot(c[:, None], c[None, :])
    edge = (m > 0) & (inner < 0.5) & (rho < 0.97)
    return torch.where(edge, tooth, torch.zeros_like(tooth))


def jitter_margin(md, generator=None):
    """训练时给颈缘线加扰动（模拟检测误差）：整体 ±0.1 mm、平滑起伏约 ±0.15 mm，随机丢掉 0–30% 的像素。"""
    import torch
    import torch.nn.functional as F

    b = len(md)
    shift = 0.1 * torch.randn(b, 1, 1, device=md.device, generator=generator)
    coarse = torch.randn(b, 1, 8, 8, device=md.device, generator=generator) * 0.15
    wave = F.interpolate(coarse, size=md.shape[-2:], mode="bicubic", align_corners=False)[:, 0]
    keep = torch.rand(md.shape, device=md.device, generator=generator) > 0.3 * torch.rand(b, 1, 1, device=md.device, generator=generator)
    return torch.where((md > 0) & keep, (md + shift + wave).clamp(min=0.31), torch.zeros_like(md))


def encode_margin(md):
    """颈缘线图 (B, S, S) mm（0 = 没有）→ (B, 2, S, S)：先往外扩 1 像素（训练时一圈细线、推理时投影的点疏密不一，扩一下两边对得上）。"""
    import torch.nn.functional as F

    return encode_tooth(F.max_pool2d(md[:, None], 3, 1, 1)[:, 0])


def build_condition(ctx, ant, has_ant):
    """ctx、ant (B, S, S) mm，has_ant (B,) → (B, 5, S, S)。"""
    import torch

    disk = torch.tensor(pixel_directions()[1], device=ctx.device).float() * 2 - 1
    a = encode_far(torch.where(has_ant.view(-1, 1, 1) > 0, ant, torch.zeros_like(ant)))
    return torch.cat([encode_far(ctx), a, disk.expand(len(ctx), 1, SIZE, SIZE)], dim=1)


def build_unet(ch: int = 64, mult=(1, 2, 4, 4), n_classes: int = 16, cond_ch: int = 5):
    """cond_ch：条件图的通道数。5 = 其余牙齿 + 对颌 + 圆盘；7 = 再加 D 输出的牙冠图（深度 + 有无），即“在 D 上补细节”的模型。"""
    import torch
    import torch.nn.functional as F

    nn = torch.nn

    class Res(nn.Module):
        def __init__(self, cin, cout, emb):
            super().__init__()
            self.n1, self.c1 = nn.GroupNorm(32, cin), nn.Conv2d(cin, cout, 3, padding=1)
            self.e = nn.Linear(emb, cout)
            self.n2, self.c2 = nn.GroupNorm(32, cout), nn.Conv2d(cout, cout, 3, padding=1)
            self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x, e):
            h = self.c1(F.silu(self.n1(x))) + self.e(e)[:, :, None, None]
            return self.c2(F.silu(self.n2(h))) + self.skip(x)

    class Attn(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.n, self.qkv, self.o = nn.GroupNorm(32, c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1)

        def forward(self, x):
            b, c, h, w = x.shape
            q, k, v = self.qkv(self.n(x)).reshape(b, 3, 4, c // 4, h * w).transpose(-1, -2).unbind(1)
            return x + self.o(F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(b, c, h, w))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            emb = ch * 4
            self.time = nn.Sequential(nn.Linear(ch, emb), nn.SiLU(), nn.Linear(emb, emb))
            self.cls = nn.Embedding(n_classes, emb)
            self.cond_ch = cond_ch
            self.inp = nn.Conv2d(2 + cond_ch, ch, 3, padding=1)
            self.down, self.up = nn.ModuleList(), nn.ModuleList()
            chans, c = [ch], ch
            for i, m in enumerate(mult):
                for _ in range(2):
                    self.down.append(nn.ModuleList([Res(c, ch * m, emb), Attn(ch * m) if i == len(mult) - 1 else nn.Identity()]))
                    c = ch * m
                    chans.append(c)
                if i < len(mult) - 1:
                    self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                    chans.append(c)
            self.mid = nn.ModuleList([Res(c, c, emb), Attn(c), Res(c, c, emb)])
            for i, m in reversed(list(enumerate(mult))):
                for _ in range(3):
                    self.up.append(nn.ModuleList([Res(c + chans.pop(), ch * m, emb), Attn(ch * m) if i == len(mult) - 1 else nn.Identity()]))
                    c = ch * m
                if i > 0:
                    self.up.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.out = nn.Sequential(nn.GroupNorm(32, c), nn.SiLU(), nn.Conv2d(c, 2, 3, padding=1))
            self.ch = ch

        def forward(self, x, cond, t, cls):
            half = self.ch // 2
            freq = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
            a = t.float()[:, None] * freq[None]
            e = self.time(torch.cat([a.sin(), a.cos()], dim=1)) + self.cls(cls)
            h = self.inp(torch.cat([x, cond], dim=1))
            skips = [h]
            for layer in self.down:
                if isinstance(layer, nn.ModuleList):
                    h = layer[1](layer[0](h, e))
                else:
                    h = layer(h)
                skips.append(h)
            h = self.mid[2](self.mid[1](self.mid[0](h, e)), e)
            for layer in self.up:
                if isinstance(layer, nn.ModuleList):
                    h = layer[1](layer[0](torch.cat([h, skips.pop()], dim=1), e))
                else:
                    h = layer(h)
            return self.out(h)

    return UNet()


def alphas(device):
    import torch

    s = torch.linspace(0, 1, STEPS + 1, device=device)
    f = torch.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
    return (f[1:] / f[0]).clamp(1e-5, 0.9999)  # 累乘的 alpha（余弦日程）


def lowpass(depth, mask, sigma: float):
    """只在有牙冠的像素里做高斯模糊（归一化卷积），边缘不会被“没有牙冠”的像素拉低。depth、mask (B, S, S)。"""
    import torch
    import torch.nn.functional as F

    r = int(3 * sigma)
    k = torch.exp(-torch.arange(-r, r + 1, device=depth.device).float() ** 2 / (2 * sigma ** 2))
    k = k / k.sum()

    def blur(x):
        x = F.conv2d(x[:, None], k.view(1, 1, 1, -1), padding=(0, r))
        return F.conv2d(x, k.view(1, 1, -1, 1), padding=(r, 0))[:, 0]

    m = mask.float()
    return blur(depth * m) / blur(m).clamp(min=1e-3)


RES_SCALE = (4.0, 0.5)  # 残差模型：扩散的是 (牙冠图 − D 的图) × 这个系数。深度差通常 ±0.1（编码后），放大到和噪声同一量级；有无的差是 0 / ±2，缩小


def res_scale(device):
    import torch

    return torch.tensor(RES_SCALE, device=device).view(1, 2, 1, 1)


def sample(model, cond, cls, steps: int = 50, generator=None, ant=None, clearance: float | None = None,
           init=None, strength: float = 1.0, keep_sigma: float | None = None, base=None,
           cls_ref=None, style_w: float = 1.0, detail_gain: float = 1.0, style_tmin: int = 0):
    """DDIM 采样。给了 ant（mm，0 = 没有）和 clearance 时，每一步都把预测的牙冠限制在“离对颌至少 clearance”以内。

    init（编码后的牙冠图，(B, 2, S, S)）：从它加噪到 strength（0–1）那一步开始去噪（SDEdit），整体形状跟着 init。
    keep_sigma（像素）：每一步把预测的低频部分换回 init 的（保留 init 的整体形状和外轮廓），只取扩散模型的细节。
    base（编码后的 D 的图）：残差模型用。模型去噪的是 (x − base) × RES_SCALE；返回的仍是牙冠图本身。
    cls_ref + style_w：风格引导。v = v(cls_ref) + style_w × (v(cls) − v(cls_ref))；cls 是设计风格、cls_ref 是天然风格时，w > 1 放大设计风格。
    detail_gain：和 keep_sigma 一起用，细节部分（预测 − 低通）乘这个倍数。"""
    import torch

    if getattr(model, "flow", False):
        return flow_sample(model, cond, cls, steps, generator, ant, clearance, init, strength, keep_sigma, cls_ref, style_w, detail_gain, style_tmin)
    ac = alphas(cond.device)
    s = res_scale(cond.device)
    to_y = (lambda x: (x - base) * s) if base is not None else (lambda x: x)
    to_x = (lambda y: base + y / s) if base is not None else (lambda y: y)
    y = torch.randn(len(cond), 2, SIZE, SIZE, device=cond.device, generator=generator)
    t_top = STEPS - 1 if init is None else max(1, int(round(strength * (STEPS - 1))))
    ts = torch.linspace(t_top, 0, max(2, int(round(steps * t_top / (STEPS - 1)))), device=cond.device).long()
    if init is not None:
        a = ac[ts[0]]
        y = a.sqrt() * to_y(init) + (1 - a).sqrt() * y
        if keep_sigma:
            init_mask = init[:, 1] > 0
            init_low = lowpass(init[:, 0], init_mask, keep_sigma)
    for i, t in enumerate(ts):
        a = ac[t]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cond.is_cuda):
            if cls_ref is not None and style_w != 1.0 and int(t) >= style_tmin:
                vv = model(torch.cat([y, y]), torch.cat([cond, cond]), t.expand(2 * len(y)), torch.cat([cls, cls_ref])).float()
                v_s, v_r = vv.chunk(2)
                v = v_r + style_w * (v_s - v_r)
            else:
                v = model(y, cond, t.expand(len(y)), cls).float()
        x0 = to_x(a.sqrt() * y - (1 - a).sqrt() * v).clamp(-1.2, 1.2)
        if init is not None and keep_sigma:
            d = x0[:, 0]
            d = torch.where(init_mask, init_low + detail_gain * (d - lowpass(d, init_mask, keep_sigma)), d)
            x0 = torch.stack([d, init[:, 1]], dim=1).clamp(-1.2, 1.2)
        if ant is not None and clearance is not None:
            limit = torch.where(ant > 0, (ant - clearance) / TOOTH_SCALE - 1, torch.full_like(ant, 9.0))
            x0 = torch.stack([torch.minimum(x0[:, 0], limit), x0[:, 1]], dim=1)
        y0 = to_y(x0)
        eps = (y - a.sqrt() * y0) / (1 - a).sqrt()
        a_next = ac[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=y.device)
        y = a_next.sqrt() * y0 + (1 - a_next).sqrt() * eps
    return to_x(y)


def flow_sample(model, cond, cls, steps=50, generator=None, ant=None, clearance=None, init=None, strength=1.0, keep_sigma=None,
                cls_ref=None, style_w=1.0, detail_gain=1.0, style_tmin=0):
    """Flow Matching 的欧拉积分：t 从 1（或 strength，从 init 加噪）走到 0。每一步先估出 x0，做低频锁 / 对颌限制，
    再按 x_t = (1−t)·x0 + t·ε 走到下一个 t（和 DDIM 同样的写法）。style_tmin 按 t×999 比较。"""
    import torch

    y = torch.randn(len(cond), 2, SIZE, SIZE, device=cond.device, generator=generator)
    t_top = 1.0 if init is None else float(strength)
    n = max(2, int(round(steps * t_top)))
    ts = torch.linspace(t_top, 0.0, n + 1, device=cond.device)
    if init is not None:
        y = (1 - t_top) * init + t_top * y
        if keep_sigma:
            init_mask = init[:, 1] > 0
            init_low = lowpass(init[:, 0], init_mask, keep_sigma)
    x0 = y
    for i in range(n):
        t, t_next = ts[i], ts[i + 1]
        tt = (t * 999).expand(len(y))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cond.is_cuda):
            if cls_ref is not None and style_w != 1.0 and float(t) * 999 >= style_tmin:
                uu = model(torch.cat([y, y]), torch.cat([cond, cond]), torch.cat([tt, tt]), torch.cat([cls, cls_ref])).float()
                u_s, u_r = uu.chunk(2)
                u = u_r + style_w * (u_s - u_r)
            else:
                u = model(y, cond, tt, cls).float()
        x0 = (y - t * u).clamp(-1.2, 1.2)
        if init is not None and keep_sigma:
            d = x0[:, 0]
            d = torch.where(init_mask, init_low + detail_gain * (d - lowpass(d, init_mask, keep_sigma)), d)
            x0 = torch.stack([d, init[:, 1]], dim=1).clamp(-1.2, 1.2)
        if ant is not None and clearance is not None:
            limit = torch.where(ant > 0, (ant - clearance) / TOOTH_SCALE - 1, torch.full_like(ant, 9.0))
            x0 = torch.stack([torch.minimum(x0[:, 0], limit), x0[:, 1]], dim=1)
        eps = (y - (1 - t) * x0) / t.clamp(min=1e-4)
        y = (1 - t_next) * x0 + t_next * eps
    return x0


def metrics(pred, tooth, ant, has_ant) -> dict:
    """pred、tooth、ant：深度图 mm（0 = 没有）。逐像素的粗指标，正式的数字看评分卡。"""
    import torch

    pm, tm = pred > 0, tooth > 0
    both = pm & tm
    mae = ((pred - tooth).abs() * both).flatten(1).sum(1) / both.flatten(1).sum(1).clamp(min=1)
    iou = both.flatten(1).sum(1) / (pm | tm).flatten(1).sum(1).clamp(min=1)
    out = {"depth_mae_mm": float(mae.mean()), "mask_iou": float(iou.mean())}
    sel = has_ant > 0
    if sel.any():
        def clear(d):  # 到对颌的最小间隙（沿视线方向；只是近似）
            gap = torch.where((d > 0) & (ant > 0), ant - d, torch.full_like(d, 9.0))
            return gap.flatten(1).min(dim=1)[0][sel]

        cp, cr = clear(pred), clear(tooth)
        out.update({"clear_med_mm": float(cp.median()), "real_clear_med_mm": float(cr.median()),
                    "pen_rate": float((cp < -0.1).float().mean()), "real_pen_rate": float((cr < -0.1).float().mean())})
    return out


def run_eval(model, data, device, n: int = 128, steps: int = 50, seed: int = 0, clearance: float | None = None) -> dict:
    import torch

    tooth, ctx, ant, has, cls = data[:5]
    dmap = data[5] if len(data) > 5 else None
    pick = np.random.default_rng(seed).choice(len(tooth), min(n, len(tooth)), replace=False)
    g = torch.Generator(device=device).manual_seed(seed)
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pick), 32):
            j = pick[i:i + 32]
            c, a, h = (torch.tensor(v[j].astype(np.float32), device=device) for v in (ctx, ant, has))
            cond, d_enc = build_condition(c, a, h), None
            if getattr(model, "margin", False):
                cond = torch.cat([cond, encode_margin(margin_edge(torch.tensor(tooth[j].astype(np.float32), device=device)))], dim=1)
            if dmap is not None:
                d_enc = encode_tooth(torch.tensor(dmap[j].astype(np.float32), device=device))
                cond = torch.cat([cond, d_enc], dim=1)
            preds.append(decode_tooth(sample(model, cond, torch.tensor(cls[j], device=device), steps, g,
                                             ant=a if clearance is not None else None, clearance=clearance,
                                             base=d_enc if getattr(model, "residual", False) else None)))
    pred = torch.cat(preds)
    out = metrics(pred, torch.tensor(tooth[pick].astype(np.float32), device=device), torch.tensor(ant[pick].astype(np.float32), device=device),
                  torch.tensor(has[pick].astype(np.float32), device=device))
    if dmap is not None:  # 对照：D 自己的逐像素误差（扩散模型补完要比它小才算有用）
        dm = metrics(torch.tensor(dmap[pick].astype(np.float32), device=device), torch.tensor(tooth[pick].astype(np.float32), device=device),
                     torch.tensor(ant[pick].astype(np.float32), device=device), torch.tensor(has[pick].astype(np.float32), device=device))
        out.update({"D_depth_mae_mm": dm["depth_mae_mm"], "D_mask_iou": dm["mask_iou"]})
    return out


def _load_init(model, path, device):
    """从已有权重开始。条件通道变多时（5 → 7，加 D 的图），输入卷积多出来的通道权重置 0：一开始的输出和原模型完全一样。"""
    import torch

    state = {k: v.float() for k, v in torch.load(path, map_location=device)["state"].items()}
    w = state["inp.weight"]
    own = model.inp.weight
    if w.shape != own.shape:
        grown = torch.zeros_like(own)
        grown[:, : w.shape[1]] = w
        state["inp.weight"] = grown
    c, own_c = state["cls.weight"], model.cls.weight
    if c.shape != own_c.shape:  # 16 → 32：设计风格的类别一开始和天然牙一样，训练里再分开
        state["cls.weight"] = c.repeat(own_c.shape[0] // c.shape[0], 1)
    model.load_state_dict(state)


def train(data_dirs, val_dir, out: Path, epochs: int, batch: int, lr: float, ant_drop: float, init: Path | None = None,
          with_d: bool = False, residual: bool = False, library: bool = False, hf_weight: float = 0.0,
          distill: Path | None = None, distill_w: float = 4.0, flow: bool = False, forever: bool = False, margin: bool = False) -> None:
    """residual（要 with_d）：扩散的对象是“真牙 − D”（× RES_SCALE），模型天然从 D 出发，只学要改的部分。"""
    assert with_d or not residual, "--residual 要和 --with-d 一起用"
    import copy

    import torch

    torch.manual_seed(0)
    device = "cuda"
    data = load_dir(data_dirs, with_d)
    tooth, ctx, ant, has, cls = data[:5]
    dmap = data[5] if with_d else None
    val = load_dir([val_dir], with_d)
    val_lib = None
    if library:  # 加上牙库造的设计风格样本（类别 + 16）
        assert not with_d, "--library 暂时只支持不带 D 的模型"
        lib = load_lib_dir(data_dirs)
        tooth, ctx, ant, has, cls = (np.concatenate([a, b]) for a, b in zip((tooth, ctx, ant, has, cls), lib))
        val_lib = load_lib_dir([val_dir])
        print(json.dumps({"library_teeth": len(lib[0]), "library_val": 0 if val_lib is None else len(val_lib[0])}), flush=True)
    print(json.dumps({"train_teeth": len(tooth), "with_antagonist": int(has.sum()), "val_teeth": len(val[0]), "with_d": with_d}), flush=True)
    assert not (margin and with_d), "--margin 暂时不和 --with-d 一起用"
    model = build_unet(cond_ch=7 if (with_d or margin) else 5, n_classes=2 * DESIGN_OFFSET if library else 16).to(device)
    model.margin = margin
    if init:
        _load_init(model, init, device)
    model.residual = residual
    model.flow = flow  # Flow Matching（直线流）：预测速度 ε − x0
    ema = copy.deepcopy(model).requires_grad_(False)
    teacher = None
    if distill:  # 老师：带设计风格类别的模型；学生的设计风格类别学“引导 distill_w 倍”以后的输出
        teacher = load_model(distill, device).requires_grad_(False)
        assert getattr(teacher, "design", False), "老师模型要有设计风格类别"
    s_res = res_scale(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ac = alphas(device)
    steps_per_epoch = len(tooth) // batch
    sched = None if forever else torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.03)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng, t0 = np.random.default_rng(0), time.time()
    start, stop = 1, {"flag": False}
    if forever:  # 长期训练：能被打断、能接着训
        import signal

        signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
        if out.exists():
            ck = torch.load(out, map_location=device, weights_only=False)
            if "opt" in ck:
                model.load_state_dict({k: v.float() for k, v in ck["raw"].items()})
                ema.load_state_dict({k: v.float() for k, v in ck["state"].items()})
                opt.load_state_dict(ck["opt"])
                start = int(ck["epoch"]) + 1
                rng = np.random.default_rng(start)
                print(json.dumps({"resume_from_epoch": int(ck["epoch"])}), flush=True)
    last_save = time.time()

    def save(epoch_done: int):
        ck = {"state": ema.state_dict(), "raw": model.state_dict(), "epoch": epoch_done, "cond_ch": model.cond_ch, "residual": residual, "flow": flow,
              "margin": margin}
        if forever:
            ck["opt"] = opt.state_dict()
        tmp = out.with_suffix(".tmp")
        torch.save(ck, tmp)
        tmp.replace(out)  # 先写临时文件再改名：存到一半被打断也不会坏掉

    for epoch in range(start, epochs + 1):
        model.train()
        order, total = rng.permutation(len(tooth)), 0.0
        for s in range(steps_per_epoch):
            j = np.sort(order[s * batch:(s + 1) * batch])
            tt = torch.tensor(tooth[j].astype(np.float32), device=device)
            x0 = encode_tooth(tt)
            h = torch.tensor(has[j].astype(np.float32), device=device)
            h = h * (torch.rand(len(j), device=device) > ant_drop)  # 一部分样本不给对颌：没有对颌时也要能用
            cond = build_condition(torch.tensor(ctx[j].astype(np.float32), device=device), torch.tensor(ant[j].astype(np.float32), device=device), h)
            if margin:  # 颈缘线：答案的外边缘 + 扰动，一部分样本不给
                md = jitter_margin(margin_edge(tt))
                md = md * (torch.rand(len(j), 1, 1, device=device) > MARGIN_DROP)
                cond = torch.cat([cond, encode_margin(md)], dim=1)
            if dmap is not None:
                d_enc = encode_tooth(jitter_d(torch.tensor(dmap[j].astype(np.float32), device=device)))
                cond = torch.cat([cond, d_enc], dim=1)
                if residual:  # 扩散的对象换成“真牙 − D”
                    x0 = (x0 - d_enc) * s_res
            t = torch.randint(0, STEPS, (len(j),), device=device)
            a = ac[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x0)
            c_j = torch.tensor(cls[j], device=device)
            if flow:  # t 取 logit-normal（中间的 t 多一些），x_t = (1−t)·x0 + t·ε，目标速度 ε − x0
                tf = torch.sigmoid(torch.randn(len(j), device=device)).view(-1, 1, 1, 1)
                xt = (1 - tf) * x0 + tf * noise
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = model(xt, cond, tf.view(-1) * 999, c_j)
                loss = ((v.float() - (noise - x0)) ** 2).mean()
                if hf_weight > 0:
                    x0_hat = xt - tf * v.float()
                    m = x0[:, 1] > 0
                    hp = lambda d: torch.where(m, d - lowpass(d, m, 3.0), torch.zeros_like(d))  # noqa: E731
                    err = (hp(x0_hat[:, 0]) - hp(x0[:, 0])) ** 2
                    loss = loss + hf_weight * ((err.sum((1, 2)) / m.sum((1, 2)).clamp(min=1)) * (1 - tf.view(-1))).mean()
            elif teacher is not None:
                xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
                nat = c_j < DESIGN_OFFSET
                c_nat, c_des = c_j % DESIGN_OFFSET, c_j % DESIGN_OFFSET + DESIGN_OFFSET
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    vv = teacher(torch.cat([xt, xt]), torch.cat([cond, cond]), torch.cat([t, t]), torch.cat([c_des, c_nat])).float()
                v_s, v_r = vv.chunk(2)
                v_guided = v_r + distill_w * (v_s - v_r)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v2 = model(torch.cat([xt, xt]), torch.cat([cond, cond]), torch.cat([t, t]), torch.cat([c_nat, c_des])).float()
                v_n, v_d = v2.chunk(2)
                target = a.sqrt() * noise - (1 - a).sqrt() * x0
                l_nat = (((v_n - target) ** 2).mean((1, 2, 3)) * nat.float()).sum() / nat.float().sum().clamp(min=1)
                loss = l_nat + ((v_d - v_guided) ** 2).mean()
                v = v_d
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = model(a.sqrt() * x0 + (1 - a).sqrt() * noise, cond, t, c_j)
                loss = ((v.float() - (a.sqrt() * noise - (1 - a).sqrt() * x0)) ** 2).mean()
            if hf_weight > 0 and not residual and not flow:  # 细节损失：还原的 x0 和答案的高频部分（窝沟、牙尖棱）对齐
                xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
                x0_hat = a.sqrt() * xt - (1 - a).sqrt() * v.float()
                m = x0[:, 1] > 0
                hp = lambda d: torch.where(m, d - lowpass(d, m, 3.0), torch.zeros_like(d))  # noqa: E731
                err = (hp(x0_hat[:, 0]) - hp(x0[:, 0])) ** 2
                w_t = a.view(-1)
                loss_hf = ((err.sum((1, 2)) / m.sum((1, 2)).clamp(min=1)) * w_t).mean()
                loss = loss + hf_weight * loss_hf
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            with torch.no_grad():
                for p, q in zip(ema.parameters(), model.parameters()):
                    p.lerp_(q, 0.001)
            total += loss.item()
            if forever and (stop["flag"] or time.time() - last_save > 900):
                save(epoch - 1)  # 只记完整的轮数：接着训时这一轮从头再来（EMA / 模型已经往前走了一点，无妨）
                last_save = time.time()
                if stop["flag"]:
                    print(json.dumps({"stopped_in_epoch": epoch, "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
                    return
        row = {"epoch": epoch, "loss": round(total / steps_per_epoch, 5), "minutes": round((time.time() - t0) / 60, 1)}
        if epoch % 5 == 0 or epoch == epochs:
            row.update({k: round(v, 4) for k, v in run_eval(ema, val, device, n=96, steps=30).items()})
            if val_lib is not None:  # 设计风格：和对齐过来的库牙比
                row.update({"design_" + k: round(v, 4) for k, v in run_eval(ema, val_lib, device, n=96, steps=30).items()})
        print(json.dumps(row), flush=True)
        save(epoch)
        last_save = time.time()
        if forever and epoch % 10 == 0:  # 每 10 轮留一个快照，之后可以挑着评估
            import shutil

            shutil.copy(out, out.with_name(f"{out.stem}.e{epoch:04d}.pt"))


def load_model(path: Path, device: str):
    import torch

    ck = torch.load(path, map_location=device)
    state = {k: v.float() for k, v in ck["state"].items()}
    model = build_unet(cond_ch=state["inp.weight"].shape[1] - 2, n_classes=state["cls.weight"].shape[0]).to(device)
    model.load_state_dict(state)
    model.design = state["cls.weight"].shape[0] > DESIGN_OFFSET  # 有设计风格的类别
    model.residual = bool(ck.get("residual", False))
    model.flow = bool(ck.get("flow", False))
    model.margin = bool(ck.get("margin", False))  # 条件里有颈缘线（多出的 2 个通道不是 D 的图）
    return model.eval()


def clean_depth(depth: np.ndarray) -> np.ndarray:
    """采样出来的深度图：只留最大的一块、去掉毛刺、补上内部的洞（网格要只有一圈开放边界，封底才做得了）。"""
    from scipy import ndimage

    valid = ndimage.binary_opening(depth > 0.3, structure=np.ones((3, 3)))
    lab, n = ndimage.label(valid)
    if n == 0:
        return np.zeros_like(depth)
    valid = lab == (np.argmax(ndimage.sum(valid, lab, range(1, n + 1))) + 1)
    filled = ndimage.binary_fill_holes(valid) & pixel_directions()[1]
    if (filled & ~valid).any():
        idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        depth = depth[idx[0], idx[1]]
    return np.where(filled, depth, 0.0)


class RadialDiffusionGenerator:
    """径向深度图扩散模型生成缺失牙，实现 ToothGenerator 接口。做不了的情况（第三磨牙、没有邻牙、网格不成形）交给 fallback。"""

    name = "radial_diffusion"

    def __init__(self, model_path, fallback=None, device: str | None = None, steps: int = 50, clearance: float | None = None, seed: int = 0,
                 refine: float | None = None, keep_sigma: float | None = None, style: str = "natural",
                 style_w: float = 1.0, detail_gain: float = 1.0, style_tmin: int = 0, smooth: float = 0.0, terminal_model=None):
        """refine（0–1）：先让 fallback（学习模型 D）出整体形状，扩散模型从它加噪到这一步再去噪，只补细节。keep_sigma 见 sample。"""
        import os

        self.model_path, self.fallback = Path(model_path), fallback
        self.device = device or os.environ.get("GEN_DEVICE")
        self.steps, self.clearance, self.seed = steps, clearance, seed
        self.refine, self.keep_sigma = refine, keep_sigma
        self.style = style  # "design"：模型训过牙库样本时，按技工设计的牙形风格生成（牙尖、沟更清楚）
        self.style_w, self.detail_gain, self.style_tmin = style_w, detail_gain, style_tmin  # 见 sample
        self.smooth = smooth  # 生成后抹平麻点的 σ（像素），0 = 不抹
        # 末端牙（只有一侧有邻牙）单独的模型：用末端牙样本训过，配外推原点；没给就和其余牙一样
        self.terminal_path, self._terminal = (Path(terminal_model) if terminal_model else None), None
        self._model = None

    @property
    def library(self):
        return getattr(self.fallback, "library", None)

    def _fall(self, ctx, target, reason: str):
        if self.fallback is None:
            raise ValueError(f"{target.fdi} 扩散模型不支持：{reason}")
        tooth = self.fallback.generate(ctx, target)
        tooth.info["radial_fallback"] = reason
        return tooth

    accepts_margin = True  # generate 可以接 margin（颈缘线 3D 点）；模型没训过颈缘线条件时忽略

    def generate(self, ctx, target, margin=None):
        import torch
        import trimesh

        from ..geometry import estimate_arch_frame
        from ..margin.cervical import boundary_loops
        from .base import GeneratedTooth
        from .learned import canonical_rotation, target_class
        from .radial import gap_origin, is_one_sided, map_to_mesh, tooth_maps

        if target.kind == "closed":
            raise ValueError(f"{target.fdi} 的间隙已关闭，不生成")
        if target.fdi % 10 == 8:
            return self._fall(ctx, target, "第三磨牙")
        rot = canonical_rotation(estimate_arch_frame(ctx.mesh, ctx.labels, ctx.jaw), ctx.jaw)
        labels = np.asarray(ctx.labels).copy()
        labels[np.isin(labels, list(ctx.excluded_donors) + [target.fdi])] = 0
        canon = trimesh.Trimesh(np.asarray(ctx.mesh.vertices) @ rot.T, ctx.mesh.faces, process=False)
        verts = np.asarray(canon.vertices)
        centroids = {int(f): verts[labels == f].mean(axis=0) for f in np.unique(labels) if f > 0 and (labels == f).sum() >= 200}
        terminal = self.terminal_path is not None and is_one_sided(centroids, target.fdi)
        origin = gap_origin(centroids, target.fdi, extrapolate=True) if terminal else gap_origin(
            centroids, target.fdi, extrapolate=False if self.terminal_path is not None else None)
        if origin is None:
            return self._fall(ctx, target, "两侧都没有现存牙")
        flip = target.fdi // 10 in (2, 3)
        ant = None
        if ctx.antagonist is not None and len(ctx.antagonist.vertices):
            ant = trimesh.Trimesh(np.asarray(ctx.antagonist.vertices) @ rot.T, ctx.antagonist.faces, process=False)
        maps = tooth_maps(canon, labels, target.fdi, origin, ant, flip=flip)
        if self._model is None:
            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = load_model(self.model_path, self.device)
        if terminal and self._terminal is None:
            self._terminal = load_model(self.terminal_path, self.device)
        model = self._terminal if terminal else self._model
        has = "ant" in maps
        a = torch.tensor(maps.get("ant", np.zeros((SIZE, SIZE), np.float32)), device=self.device)[None]
        cond = build_condition(torch.tensor(maps["ctx"], device=self.device)[None], a, torch.tensor([float(has)], device=self.device))
        g = torch.Generator(device=self.device).manual_seed(self.seed)
        init, base = None, None
        use_margin = getattr(model, "margin", False)
        if use_margin:  # 颈缘线投到同一张径向图上；没给颈缘线就是“没有”（训练时 15% 的样本也这样）
            from .radial import splat

            md = np.zeros((SIZE, SIZE), np.float32)
            if margin is not None and len(margin) >= 8:
                from ..margin.detect import resample_loop

                sign_m = np.array([-1.0, 1.0, 1.0]) if flip else np.ones(3)
                pts = resample_loop(np.asarray(margin, float), 4000) @ rot.T
                md = splat((pts - origin) * sign_m, np.zeros(3), fill=0)
            cond = torch.cat([cond, encode_margin(torch.tensor(md, device=self.device).float()[None])], dim=1)
        need_d = model.cond_ch > (7 if use_margin else 5)  # 专门“在 D 上补细节”训练的模型：D 的图是条件之一
        if need_d and self.fallback is None:
            raise ValueError("这个扩散模型要 D 的输出当条件，需要 fallback=LearnedGenerator")
        if (self.refine is not None or need_d) and self.fallback is not None:  # 学习模型 D 出整体形状，投到同一张径向图上
            from .radial import splat

            if margin is not None and getattr(self.fallback, "accepts_margin", False):  # 形状模型也看颈缘线（训过颈缘线条件的才用）
                base = self.fallback.generate(ctx, target, margin=margin)
            else:
                base = self.fallback.generate(ctx, target)
            sign0 = np.array([-1.0, 1.0, 1.0]) if flip else np.ones(3)
            pts = np.asarray(trimesh.sample.sample_surface(base.crown, 150000, seed=0)[0]) @ rot.T
            d0 = clean_depth(splat((pts - origin) * sign0, np.zeros(3)))
            if (d0 > 0).sum() < 1500:
                base.info["radial_fallback"] = "D 的牙冠投不到径向图上"
                return base
            d_enc = encode_tooth(torch.tensor(d0, device=self.device).float()[None])
            init = d_enc if self.refine is not None else None
            if need_d:
                cond = torch.cat([cond, d_enc], dim=1)
        with torch.no_grad():
            design = self.style == "design" and getattr(model, "design", False)
            cls = target_class(target.fdi) + (DESIGN_OFFSET if design else 0)
            x = sample(model, cond, torch.tensor([cls], device=self.device), self.steps, g,
                       ant=a if has and self.clearance is not None else None, clearance=self.clearance,
                       init=init, strength=self.refine or 1.0, keep_sigma=self.keep_sigma,
                       base=d_enc if need_d and getattr(model, "residual", False) else None,
                       cls_ref=torch.tensor([target_class(target.fdi)], device=self.device) if design else None,
                       style_w=self.style_w, detail_gain=self.detail_gain, style_tmin=self.style_tmin)
        dec = decode_tooth(x)
        if self.smooth > 0:
            dec = torch.where(dec > 0, lowpass(dec, dec > 0, self.smooth), dec)
        depth = clean_depth(dec[0].cpu().numpy())
        if (depth > 0).sum() < 1500:
            if base is not None:
                base.info["radial_fallback"] = "生成的牙冠太小"
                return base
            return self._fall(ctx, target, "生成的牙冠太小")
        crown = map_to_mesh(depth, np.zeros(3))
        sign = np.array([-1.0, 1.0, 1.0]) if flip else np.ones(3)
        world = (np.asarray(crown.vertices) * sign + origin) @ rot
        crown = trimesh.Trimesh(world, crown.faces[:, ::-1] if flip else crown.faces, process=False)
        if ((crown.triangles_center - world.mean(axis=0)) * crown.face_normals).sum() < 0:
            crown.invert()
        if len(boundary_loops(crown)) != 1:
            if base is not None:
                base.info["radial_fallback"] = "生成的牙冠边界不是一圈"
                return base
            return self._fall(ctx, target, "生成的牙冠边界不是一圈")
        info = {"source": f"radial_diffusion:{(self.terminal_path if terminal else self.model_path).name}", "terminal": terminal, "placement": "radial_diffusion", "gap_mm": target.gap_mm,
                "occlusal_drop_mm": 0.0, "antagonist_input": has}
        if use_margin:
            info["margin_condition"] = margin is not None
        if base is not None:
            info.update({k: v for k, v in base.info.items() if k not in ("source", "placement", "occlusal_drop_mm")})
            kind = "radial_dcond" if need_d else "radial_refine"
            info.update({"source": f"{kind}:{self.model_path.name}", "refine": self.refine, "keep_sigma": self.keep_sigma})
        info["style"] = "design" if self.style == "design" and getattr(model, "design", False) else "natural"
        if info["style"] == "design" and self.style_w != 1.0:
            info["style_w"] = self.style_w
            if self.style_tmin:
                info["style_tmin"] = self.style_tmin
        if self.smooth > 0:
            info["smooth"] = self.smooth
        if self.detail_gain != 1.0:
            info["detail_gain"] = self.detail_gain
        return GeneratedTooth(fdi=target.fdi, crown=crown, method=self.name, info=info)


def main() -> None:
    p = argparse.ArgumentParser(description="径向深度图上的牙冠扩散模型")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, nargs="+", required=True)
    t.add_argument("--val", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--ant-drop", type=float, default=0.2)
    t.add_argument("--init", type=Path)
    t.add_argument("--with-d", action="store_true", help="条件里加 D 的输出（data/radial_d/<split>，radial_dmaps.py 生成）：专门在 D 上补细节")
    t.add_argument("--residual", action="store_true", help="和 --with-d 一起：扩散“真牙 − D”，模型从 D 出发只学要改的部分")
    t.add_argument("--hf-weight", type=float, default=0.0, help="细节损失的权重（0 = 不加）")
    t.add_argument("--distill", type=Path, help="引导蒸馏：老师模型（有设计风格类别）")
    t.add_argument("--distill-w", type=float, default=4.0, help="蒸馏的引导倍数")
    t.add_argument("--flow", action="store_true", help="Flow Matching（直线流）代替扩散")
    t.add_argument("--forever", action="store_true", help="长期训练：学习率固定、能被 SIGTERM 打断、out 存在就接着训")
    t.add_argument("--margin", action="store_true", help="条件里加颈缘线（预备体牙冠外表面用）：训练时从答案的外边缘取")
    t.add_argument("--library", action="store_true", help="加上牙库造的设计风格样本（data/radial_lib/<split>，radial_library.py build 生成）")
    e = sub.add_parser("eval")
    e.add_argument("--model", type=Path, required=True)
    e.add_argument("--val", type=Path, required=True)
    e.add_argument("--n", type=int, default=256)
    e.add_argument("--steps", type=int, default=50)
    e.add_argument("--clearance", type=float)
    args = p.parse_args()
    if args.cmd == "train":
        train(args.data, args.val, args.out, args.epochs, args.batch, args.lr, args.ant_drop, args.init, args.with_d, args.residual, args.library, args.hf_weight,
              args.distill, args.distill_w, args.flow, args.forever, args.margin)
    else:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(json.dumps(run_eval(load_model(args.model, device), load_dir([args.val]), device, args.n, args.steps, clearance=args.clearance)))


if __name__ == "__main__":
    main()
