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
import math
import time
from pathlib import Path

import numpy as np

from .radial import SIZE, pixel_directions

TOOTH_SCALE = 5.0  # 牙冠深度：r / 5 − 1（牙冠半径 1.5–9 mm → −0.7–0.8）；没有牙冠的像素 −1
FAR_SCALE = 8.0  # 上下文 / 对颌：r / 8 − 1（最远 16 mm → 1）；没有东西的像素 1（“很远”）
STEPS = 1000


def load_dir(dirs: list[Path]):
    from .learned import target_class

    tooth, ctx, ant, has, cls = [], [], [], [], []
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            with np.load(f) as z:
                tooth.append(z["tooth"]), ctx.append(z["ctx"]), ant.append(z["ant"]), has.append(z["has_ant"])
                cls.append(np.array([target_class(int(v)) for v in z["fdi"]], np.int64))
    return np.concatenate(tooth), np.concatenate(ctx), np.concatenate(ant), np.concatenate(has), np.concatenate(cls)


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


def build_condition(ctx, ant, has_ant):
    """ctx、ant (B, S, S) mm，has_ant (B,) → (B, 5, S, S)。"""
    import torch

    disk = torch.tensor(pixel_directions()[1], device=ctx.device).float() * 2 - 1
    a = encode_far(torch.where(has_ant.view(-1, 1, 1) > 0, ant, torch.zeros_like(ant)))
    return torch.cat([encode_far(ctx), a, disk.expand(len(ctx), 1, SIZE, SIZE)], dim=1)


def build_unet(ch: int = 64, mult=(1, 2, 4, 4), n_classes: int = 16):
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
            self.inp = nn.Conv2d(2 + 5, ch, 3, padding=1)
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


def sample(model, cond, cls, steps: int = 50, generator=None, ant=None, clearance: float | None = None):
    """DDIM 采样。给了 ant（mm，0 = 没有）和 clearance 时，每一步都把预测的牙冠限制在“离对颌至少 clearance”以内。"""
    import torch

    ac = alphas(cond.device)
    x = torch.randn(len(cond), 2, SIZE, SIZE, device=cond.device, generator=generator)
    ts = torch.linspace(STEPS - 1, 0, steps, device=cond.device).long()
    for i, t in enumerate(ts):
        a = ac[t]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cond.is_cuda):
            v = model(x, cond, t.expand(len(x)), cls).float()
        x0 = (a.sqrt() * x - (1 - a).sqrt() * v).clamp(-1.2, 1.2)
        if ant is not None and clearance is not None:
            limit = torch.where(ant > 0, (ant - clearance) / TOOTH_SCALE - 1, torch.full_like(ant, 9.0))
            x0 = torch.stack([torch.minimum(x0[:, 0], limit), x0[:, 1]], dim=1)
        eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
        a_next = ac[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=x.device)
        x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
    return x


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

    tooth, ctx, ant, has, cls = data
    pick = np.random.default_rng(seed).choice(len(tooth), min(n, len(tooth)), replace=False)
    g = torch.Generator(device=device).manual_seed(seed)
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pick), 32):
            j = pick[i:i + 32]
            c, a, h = (torch.tensor(v[j].astype(np.float32), device=device) for v in (ctx, ant, has))
            preds.append(decode_tooth(sample(model, build_condition(c, a, h), torch.tensor(cls[j], device=device), steps, g,
                                             ant=a if clearance is not None else None, clearance=clearance)))
    pred = torch.cat(preds)
    return metrics(pred, torch.tensor(tooth[pick].astype(np.float32), device=device), torch.tensor(ant[pick].astype(np.float32), device=device),
                   torch.tensor(has[pick].astype(np.float32), device=device))


def train(data_dirs, val_dir, out: Path, epochs: int, batch: int, lr: float, ant_drop: float, init: Path | None = None) -> None:
    import copy

    import torch

    torch.manual_seed(0)
    device = "cuda"
    tooth, ctx, ant, has, cls = load_dir(data_dirs)
    val = load_dir([val_dir])
    print(json.dumps({"train_teeth": len(tooth), "with_antagonist": int(has.sum()), "val_teeth": len(val[0])}), flush=True)
    model = build_unet().to(device)
    if init:
        model.load_state_dict(torch.load(init, map_location=device)["state"])
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ac = alphas(device)
    steps_per_epoch = len(tooth) // batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.03)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng, t0 = np.random.default_rng(0), time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        order, total = rng.permutation(len(tooth)), 0.0
        for s in range(steps_per_epoch):
            j = np.sort(order[s * batch:(s + 1) * batch])
            x0 = encode_tooth(torch.tensor(tooth[j].astype(np.float32), device=device))
            h = torch.tensor(has[j].astype(np.float32), device=device)
            h = h * (torch.rand(len(j), device=device) > ant_drop)  # 一部分样本不给对颌：没有对颌时也要能用
            cond = build_condition(torch.tensor(ctx[j].astype(np.float32), device=device), torch.tensor(ant[j].astype(np.float32), device=device), h)
            t = torch.randint(0, STEPS, (len(j),), device=device)
            a = ac[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(a.sqrt() * x0 + (1 - a).sqrt() * noise, cond, t, torch.tensor(cls[j], device=device))
            loss = ((v.float() - (a.sqrt() * noise - (1 - a).sqrt() * x0)) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for p, q in zip(ema.parameters(), model.parameters()):
                    p.lerp_(q, 0.001)
            total += loss.item()
        row = {"epoch": epoch, "loss": round(total / steps_per_epoch, 5), "minutes": round((time.time() - t0) / 60, 1)}
        if epoch % 5 == 0 or epoch == epochs:
            row.update({k: round(v, 4) for k, v in run_eval(ema, val, device, n=96, steps=30).items()})
        print(json.dumps(row), flush=True)
        torch.save({"state": ema.state_dict(), "raw": model.state_dict(), "epoch": epoch}, out)


def load_model(path: Path, device: str):
    import torch

    model = build_unet().to(device)
    model.load_state_dict(torch.load(path, map_location=device)["state"])
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

    def __init__(self, model_path, fallback=None, device: str | None = None, steps: int = 50, clearance: float | None = None, seed: int = 0):
        import os

        self.model_path, self.fallback = Path(model_path), fallback
        self.device = device or os.environ.get("GEN_DEVICE")
        self.steps, self.clearance, self.seed = steps, clearance, seed
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

    def generate(self, ctx, target):
        import torch
        import trimesh

        from ..geometry import estimate_arch_frame
        from ..margin.cervical import boundary_loops
        from .base import GeneratedTooth
        from .learned import canonical_rotation, target_class
        from .radial import gap_origin, map_to_mesh, tooth_maps

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
        origin = gap_origin(centroids, target.fdi)
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
        has = "ant" in maps
        a = torch.tensor(maps.get("ant", np.zeros((SIZE, SIZE), np.float32)), device=self.device)[None]
        cond = build_condition(torch.tensor(maps["ctx"], device=self.device)[None], a, torch.tensor([float(has)], device=self.device))
        g = torch.Generator(device=self.device).manual_seed(self.seed)
        with torch.no_grad():
            x = sample(self._model, cond, torch.tensor([target_class(target.fdi)], device=self.device), self.steps, g,
                       ant=a if has and self.clearance is not None else None, clearance=self.clearance)
        depth = clean_depth(decode_tooth(x)[0].cpu().numpy())
        if (depth > 0).sum() < 1500:
            return self._fall(ctx, target, "生成的牙冠太小")
        crown = map_to_mesh(depth, np.zeros(3))
        sign = np.array([-1.0, 1.0, 1.0]) if flip else np.ones(3)
        world = (np.asarray(crown.vertices) * sign + origin) @ rot
        crown = trimesh.Trimesh(world, crown.faces[:, ::-1] if flip else crown.faces, process=False)
        if ((crown.triangles_center - world.mean(axis=0)) * crown.face_normals).sum() < 0:
            crown.invert()
        if len(boundary_loops(crown)) != 1:
            return self._fall(ctx, target, "生成的牙冠边界不是一圈")
        info = {"source": f"radial_diffusion:{self.model_path.name}", "placement": "radial_diffusion", "gap_mm": target.gap_mm,
                "occlusal_drop_mm": 0.0, "antagonist_input": has}
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
    e = sub.add_parser("eval")
    e.add_argument("--model", type=Path, required=True)
    e.add_argument("--val", type=Path, required=True)
    e.add_argument("--n", type=int, default=256)
    e.add_argument("--steps", type=int, default=50)
    e.add_argument("--clearance", type=float)
    args = p.parse_args()
    if args.cmd == "train":
        train(args.data, args.val, args.out, args.epochs, args.batch, args.lr, args.ant_drop, args.init)
    else:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(json.dumps(run_eval(load_model(args.model, device), load_dir([args.val]), device, args.n, args.steps, clearance=args.clearance)))


if __name__ == "__main__":
    main()
