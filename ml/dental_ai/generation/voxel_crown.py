"""体素牙冠模型（照 VBCD 的结构）：预备体、邻牙、对颌放进 128³ 体素（20 mm 立方体、0.156 mm 一格），
3D 残差 U-Net 出牙冠表面占据 + 法向，表面体素再用逐点精修器出亚体素偏移和法向（曲率 / 颈缘加权的倒角损失 CMPL）。

和 VBCD 原版的区别（原版代码不完整，这些是我们自己定的）：
- 立方体中心用预备体包围盒中心（+ 训练集平均的牙冠偏移），不用技师牙冠的包围盒（那是真值，推理时没有）；
- 输入 3 个通道（预备体 / 邻牙 / 对颌）分开，不是混在一个通道；
- 体素法向损失只在真值表面体素上算；
- 曲率 = 网格顶点法向和邻居的偏差 / 边长，颈缘 = 离牙冠开口边 0.5 mm 以内。

数据（FDI 16 Crown Dataset）坐标：y 朝咬合面，单位 mm。

    python -m dental_ai.generation.voxel_crown cache --data ~/zyt/dental_data/crownfusion/CrownFusion --split train --out ../data/vox_cache/train
    python -m dental_ai.generation.voxel_crown train --cache ../data/vox_cache/train --val ../data/vox_cache/val --out ../runs/voxel/vox.pt
    python -m dental_ai.generation.voxel_crown eval --data ... --split test --model ../runs/voxel/vox.pt --out ../runs/voxel/cf_test
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

GRID, VOX = 128, 0.15625  # 128 × 0.15625 = 20 mm
HALF = GRID * VOX / 2
Y_OFF = 1.26  # 训练集：技师牙冠包围盒中心比预备体包围盒中心高 1.26 mm（y）
N_CROWN, N_GT, N_PREP, N_CTX = 60000, 8192, 40000, 30000


# ---------------------------------------------------------------- 缓存
def _cache_case(args):
    import trimesh
    from scipy.spatial import cKDTree

    from ..eval_crownfusion import load_case, rim_points

    d, out = args
    dest = out / f"{d.name}.npz"
    if dest.exists():
        return True
    try:
        pts, props, prep, crown = load_case(d)
        seed = int(d.name)
        p, fi = trimesh.sample.sample_surface(crown, N_CROWN, seed=seed)
        nrm = np.asarray(crown.face_normals)[fi]
        if ((p - p.mean(0)) * nrm).sum(1).mean() < 0:
            nrm = -nrm
        vn = np.asarray(crown.vertex_normals)
        e, ln = crown.edges_unique, crown.edges_unique_length
        val = (1 - (vn[e[:, 0]] * vn[e[:, 1]]).sum(1)) / np.maximum(ln, 1e-3)
        n = len(vn)
        k = (np.bincount(e[:, 0], val, n) + np.bincount(e[:, 1], val, n)) / np.maximum(
            np.bincount(e[:, 0], None, n) + np.bincount(e[:, 1], None, n), 1)
        kp = k[np.asarray(crown.faces)[fi]].mean(1)
        rim = rim_points(crown)
        mg = (cKDTree(rim).query(p)[0] < 0.5) if len(rim) else np.zeros(len(p), bool)
        pp, _ = trimesh.sample.sample_surface(prep, N_PREP, seed=seed + 1)
        tooth = props[:, 1] == 0
        np.savez(dest, crown=np.concatenate([p, nrm, kp[:, None], mg[:, None]], 1).astype(np.float16),
                 prep=pp.astype(np.float32), ctx=pts[:, :3].astype(np.float32), ctx_type=(~tooth).astype(np.int8))
        return True
    except Exception as exc:  # noqa: BLE001
        print("skip", d.name, type(exc).__name__, exc, flush=True)
        return False


def build_cache(data: Path, split: str, out: Path, workers: int) -> int:
    import multiprocessing as mp

    ids = json.load(open(data / {"test": "testing.json", "val": "validation.json", "train": "training.json"}[split]))
    out.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        return sum(pool.imap_unordered(_cache_case, [(data / split / str(c), out) for c in ids], chunksize=8))


def crop_center(prep: np.ndarray) -> np.ndarray:
    return (prep.min(0) + prep.max(0)) / 2 + np.array([0, Y_OFF, 0])


class VoxDataset:
    def __init__(self, files, train: bool):
        self.files, self.train = list(files), train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import torch

        z = np.load(self.files[i])
        rng = np.random.default_rng() if self.train else np.random.default_rng(i)
        prep, ctx, ct, crown = z["prep"], z["ctx"], z["ctx_type"], z["crown"].astype(np.float32)
        c = crop_center(prep)
        if self.train:
            c = c + rng.normal(0, 0.4, 3)  # 中心抖动，推理时预备体估的中心也会偏
        gt = crown[rng.choice(len(crown), N_GT, replace=False)]
        ctx6 = np.zeros((N_CTX, 4), np.float32)
        m = min(len(ctx), N_CTX)
        sel = rng.choice(len(ctx), m, replace=False)
        ctx6[:m, :3], ctx6[:m, 3] = ctx[sel] - c, ct[sel] + 1  # 1 = 邻牙、2 = 对颌、0 = 空
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))  # noqa: E731
        return {"prep": t(prep - c), "ctx": t(ctx6), "crown": t(crown[:, :3] - c), "crown_n": t(crown[:, 3:6]),
                "gt": t(np.concatenate([gt[:, :3] - c, gt[:, 3:]], 1)), "center": t(c)}


# ---------------------------------------------------------------- 体素化（GPU）
def voxelize(points, channel, b_idx, B, C, device):
    """points (M,3) mm（立方体中心为原点）、channel (M,) 通道、b_idx (M,) 样本 → (B,C,G,G,G) 0/1。"""
    import torch

    ijk = torch.floor((points + HALF) / VOX).long()
    ok = ((ijk >= 0) & (ijk < GRID)).all(1)
    ijk, channel, b_idx = ijk[ok], channel[ok], b_idx[ok]
    g = torch.zeros(B * C * GRID ** 3, device=device)
    flat = (((b_idx * C + channel) * GRID + ijk[:, 0]) * GRID + ijk[:, 1]) * GRID + ijk[:, 2]
    g[flat] = 1
    return g.view(B, C, GRID, GRID, GRID)


def make_inputs(batch, device):
    import torch

    prep, ctx = batch["prep"].to(device), batch["ctx"].to(device)
    B = prep.shape[0]
    bp = torch.arange(B, device=device).repeat_interleave(prep.shape[1])
    bc = torch.arange(B, device=device).repeat_interleave(ctx.shape[1])
    t = ctx[..., 3].reshape(-1).long()
    keep = t > 0
    pts = torch.cat([prep.reshape(-1, 3), ctx[..., :3].reshape(-1, 3)[keep]])
    ch = torch.cat([torch.zeros(len(bp), dtype=torch.long, device=device), t[keep]])
    return voxelize(pts, ch, torch.cat([bp, bc[keep]]), B, 3, device)


def make_targets(batch, device):
    """真值：表面占据 (B,1,G³) + 每个表面体素的平均法向 (B,3,G³)。"""
    import torch

    crown, cn = batch["crown"].to(device), batch["crown_n"].to(device)
    B, M = crown.shape[:2]
    ijk = torch.floor((crown.reshape(-1, 3) + HALF) / VOX).long()
    b = torch.arange(B, device=device).repeat_interleave(M)
    ok = ((ijk >= 0) & (ijk < GRID)).all(1)
    flat = ((b * GRID + ijk[:, 0]) * GRID + ijk[:, 1]) * GRID + ijk[:, 2]
    flat, nn_ = flat[ok], cn.reshape(-1, 3)[ok]
    occ = torch.zeros(B * GRID ** 3, device=device)
    occ[flat] = 1
    nsum = torch.zeros(B * GRID ** 3, 3, device=device).index_add_(0, flat, nn_)
    nsum = torch.nn.functional.normalize(nsum, dim=1)
    return occ.view(B, 1, GRID, GRID, GRID), nsum.view(B, GRID, GRID, GRID, 3).permute(0, 4, 1, 2, 3)


# ---------------------------------------------------------------- 网络
def build_model(f0: int = 32, levels: int = 5):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class RB(nn.Module):  # VBCD 的 ResNetBlock，顺序 GroupNorm → Conv → ReLU
        def __init__(self, cin, cout):
            super().__init__()
            self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()
            self.n1, self.c1 = nn.GroupNorm(8, cout), nn.Conv3d(cout, cout, 3, padding=1, bias=False)
            self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv3d(cout, cout, 3, padding=1, bias=False)

        def forward(self, x):
            x = self.skip(x)
            h = F.relu(self.c1(self.n1(x)))
            return F.relu(self.c2(self.n2(h)) + x)

    class VoxCrown(nn.Module):
        def __init__(self):
            super().__init__()
            fm = [f0 * 2 ** i for i in range(levels)]
            self.fm = fm
            self.enc = nn.ModuleList([RB(3 if i == 0 else fm[i - 1], fm[i]) for i in range(levels)])
            self.up = nn.ModuleList([nn.ConvTranspose3d(fm[i], fm[i - 1], 3, stride=2, padding=1, output_padding=1, bias=False)
                                     for i in range(levels - 1, 0, -1)])
            self.dec = nn.ModuleList([RB(fm[i - 1], fm[i - 1]) for i in range(levels - 1, 0, -1)])
            self.head = nn.Conv3d(fm[0], 4, 1)  # 占据 logit + 法向
            self.refine = nn.Sequential(nn.Linear(fm[0] + 4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 6))
            nn.init.zeros_(self.refine[-1].weight)
            nn.init.zeros_(self.refine[-1].bias)

        def forward(self, x):
            skips = []
            for i, e in enumerate(self.enc):
                if i:
                    x = F.max_pool3d(x, 2)
                x = e(x)
                skips.append(x)
            for j, (u, d) in enumerate(zip(self.up, self.dec)):
                x = d(u(x) + skips[-2 - j])  # VBCD：跳连相加
            return self.head(x), x

        def refine_points(self, feat, logit, idx_list):
            """idx_list：每个样本的表面体素下标 (N,3) → 精修后的点 (N,3) mm（立方体中心为原点）、法向 (N,3)。"""
            out = []
            for b, ijk in enumerate(idx_list):
                f = feat[b, :, ijk[:, 0], ijk[:, 1], ijk[:, 2]].T.float()
                lg = logit[b, :, ijk[:, 0], ijk[:, 1], ijk[:, 2]].T.float()
                o = self.refine(torch.cat([f, lg], 1))
                pos = (ijk.float() + 0.5) * VOX - HALF + 2 * VOX * torch.tanh(o[:, :3])  # 偏移最多 2 格
                out.append((pos, F.normalize(o[:, 3:] + lg[:, 1:4], dim=1)))
            return out

    return VoxCrown()


def surface_voxels(logit, cap: int | None, gen=None):
    """logit (B,4,G³) → 每个样本 sigmoid > 0.5 的体素下标；cap 限制个数（随机取）。"""
    import torch

    out = []
    for b in range(logit.shape[0]):
        ijk = (logit[b, 0] > 0).nonzero()
        if len(ijk) == 0:  # 一个都没有：取最大的 1000 个
            flat = logit[b, 0].flatten().topk(1000)[1]
            ijk = torch.stack([flat // GRID ** 2, flat // GRID % GRID, flat % GRID], 1)
        if cap and len(ijk) > cap:
            ijk = ijk[torch.randperm(len(ijk), device=ijk.device, generator=gen)[:cap]]
        out.append(ijk)
    return out


def nn_dist(a, b, chunk: int = 4096):
    """a (N,3)、b (M,3) → a 每个点到 b 的最近距离平方、下标。"""
    import torch

    ds, ids = [], []
    for s in range(0, len(a), chunk):
        d = torch.cdist(a[s:s + chunk], b)
        m, i = d.min(1)
        ds.append(m ** 2)
        ids.append(i)
    return torch.cat(ds), torch.cat(ids)


def cmpl(pos, nrm, gt):
    """VBCD 的 CMPL：双向最近点距离平方，按 exp(|曲率|/平均) + 颈缘 加权；外加法向 MSE。gt (M,8)：xyz、法向、曲率、颈缘。"""
    import torch

    k = gt[:, 6].abs()
    w = torch.exp((k / k.mean().clamp(min=1e-6)).clamp(max=4)) + gt[:, 7]
    d_gp, _ = nn_dist(gt[:, :3], pos)
    d_pg, i = nn_dist(pos, gt[:, :3])
    loss = (w * d_gp).sum() / w.sum() + (w[i] * d_pg).sum() / w[i].sum()
    nl = ((nrm - gt[i, 3:6]) ** 2).sum(1).mean()
    return loss, nl, (d_gp.sqrt().mean() + d_pg.sqrt().mean()) / 2


# ---------------------------------------------------------------- 训练
def train(cache: list[Path], val: Path | None, out: Path, epochs: int, batch: int, lr: float, f0: int,
          refine_from: int, workers: int, init: Path | None, max_val: int) -> None:
    import torch
    import torch.nn.functional as F

    device = "cuda"
    files = sorted(f for c in cache for f in c.glob("*.npz"))
    vfiles = sorted(val.glob("*.npz"))[:max_val] if val else []
    dl = torch.utils.data.DataLoader(VoxDataset(files, True), batch_size=batch, shuffle=True, num_workers=workers,
                                     drop_last=True, persistent_workers=True, pin_memory=True)
    vdl = torch.utils.data.DataLoader(VoxDataset(vfiles, False), batch_size=batch, num_workers=workers) if vfiles else None
    model = build_model(f0).to(device)
    if init:
        model.load_state_dict(torch.load(init, map_location=device)["state"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    pw = torch.tensor(10.0, device=device)
    best = float("inf")
    print(json.dumps({"train": len(files), "val": len(vfiles), "params_M": sum(p.numel() for p in model.parameters()) / 1e6}), flush=True)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        use_ref = ep > refine_from
        agg = {"bce": 0.0, "nrm": 0.0, "cmpl": 0.0, "pn": 0.0, "cd": 0.0}
        for it, bt in enumerate(dl):
            with torch.no_grad():
                x = make_inputs(bt, device)
                occ, tn = make_targets(bt, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit, feat = model(x)
            logit = logit.float()
            bce = F.binary_cross_entropy_with_logits(logit[:, :1], occ, pos_weight=pw)
            m = occ[:, 0] > 0
            nrm = ((logit[:, 1:4] - tn) ** 2).sum(1)[m].mean()
            loss = bce + nrm
            if use_ref:
                gt = bt["gt"].to(device)
                ref = model.refine_points(feat, logit, surface_voxels(logit.detach(), 12000))
                cl, pn, cdm = zip(*[cmpl(p, n, gt[b]) for b, (p, n) in enumerate(ref)])
                cl, pn = torch.stack(cl).mean(), torch.stack(pn).mean()
                loss = loss + 10 * cl + 0.1 * pn  # 距离平方 mm² 量级小，放大
                agg["cmpl"] += cl.item(); agg["pn"] += pn.item(); agg["cd"] += torch.stack(cdm).mean().item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            agg["bce"] += bce.item(); agg["nrm"] += nrm.item()
            if it % 200 == 0:
                print(json.dumps({"epoch": ep, "it": it, "bce": round(bce.item(), 4), "min": round((time.time() - t0) / 60, 1)}), flush=True)
        rec = {"epoch": ep, **{k: round(v / len(dl), 4) for k, v in agg.items()}, "minutes": round((time.time() - t0) / 60, 1)}
        if vdl is not None:
            rec.update(validate(model, vdl, device))
        print(json.dumps(rec), flush=True)
        state = {"state": model.state_dict(), "f0": f0, "epoch": ep, "val": rec}
        torch.save(state, out.with_suffix(".last.pt"))
        score = rec.get("val_cd_l2", rec["cd"] or rec["bce"])
        if use_ref and score < best:
            best = score
            torch.save(state, out)


def predict_points(model, x, n_out: int | None = None):
    """输入体素 → 每个样本精修后的点、法向（立方体坐标）。"""
    import torch

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logit, feat = model(x)
    with torch.no_grad():
        return model.refine_points(feat, logit.float(), surface_voxels(logit.float(), None))


def validate(model, vdl, device):
    import torch

    model.eval()
    cds, cfs, ns = [], [], []
    g = torch.Generator(device=device).manual_seed(0)
    for bt in vdl:
        x = make_inputs(bt, device)
        crown = bt["crown"].to(device)
        for b, (p, _) in enumerate(predict_points(model, x)):
            ns.append(len(p))
            a = p[torch.randperm(len(p), device=device, generator=g)[:10000]]
            t = crown[b][torch.randperm(crown.shape[1], device=device, generator=g)[:10000]]
            d1, _ = nn_dist(a, t)
            d2, _ = nn_dist(t, a)
            cfs.append((d1.mean() + d2.mean()).sqrt().item())
            cds.append(((d1.sqrt().mean() + d2.sqrt().mean()) / 2).item())
    model.train()
    return {"val_cd_l2": round(float(np.mean(cfs)), 4), "val_cd": round(float(np.mean(cds)), 4), "val_pts": int(np.mean(ns))}


def load(path: Path, device="cuda"):
    import torch

    ck = torch.load(path, map_location=device)
    m = build_model(ck.get("f0", 32)).to(device)
    m.load_state_dict(ck["state"])
    return m.eval()


def evaluate(data: Path, split: str, model_path: Path, out: Path, limit: int | None) -> None:
    """真实预备体测试集：和 eval_crownfusion 一样的 CD-L2（各 1 万点：技师牙冠网格表面采样 vs 生成的点）。"""
    import torch
    from scipy.spatial import cKDTree

    from ..eval_crownfusion import load_case, surface_points

    model = load(model_path)
    ids = json.load(open(data / {"test": "testing.json", "val": "validation.json", "train": "training.json"}[split]))[:limit]
    out.mkdir(parents=True, exist_ok=True)
    rows, t_inf = [], []
    for k, cid in enumerate(ids):
        d = data / split / str(cid)
        pts, props, prep, crown = load_case(d)
        pp = surface_points(prep, N_PREP, int(cid) + 1)
        c = crop_center(pp)
        bt = {"prep": torch.tensor(pp - c, dtype=torch.float32)[None],
              "ctx": torch.tensor(np.concatenate([pts[:, :3] - c, (props[:, 1:2] + 1)], 1), dtype=torch.float32)[None]}
        torch.cuda.synchronize(); t = time.time()
        p, n = predict_points(model, make_inputs(bt, "cuda"))[0]
        torch.cuda.synchronize(); t_inf.append(time.time() - t)
        p = p.cpu().numpy() + c
        rng = np.random.default_rng(1)
        a10 = p[rng.choice(len(p), 10000, replace=len(p) < 10000)]
        b10 = surface_points(crown, 10000, 1)
        da, db = cKDTree(b10).query(a10)[0], cKDTree(a10).query(b10)[0]
        rows.append({"case": str(cid), "cd_l2_cf": float(np.sqrt((da ** 2).mean() + (db ** 2).mean())),
                     "cd": float((da.mean() + db.mean()) / 2), "gen_to_gt": float(da.mean()), "gt_to_gen": float(db.mean()),
                     "height_err": float(p[:, 1].max() - np.asarray(crown.vertices)[:, 1].max()), "n_pts": len(p)})
        if k < 12:
            np.save(out / f"{cid}_pts.npy", np.concatenate([p, n.cpu().numpy()], 1).astype(np.float32))
    json.dump(rows, open(out / "rows.json", "w"))
    s = {"n": len(rows), **{k: {"median": round(float(np.median([r[k] for r in rows])), 3), "mean": round(float(np.mean([r[k] for r in rows])), 3)}
                           for k in ("cd_l2_cf", "cd", "gen_to_gt", "gt_to_gen", "height_err", "n_pts")},
         "ms_per_case": round(1000 * float(np.median(t_inf[3:] or t_inf)), 1)}
    json.dump(s, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(s, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--data", type=Path, required=True); c.add_argument("--split", default="train")
    c.add_argument("--out", type=Path, required=True); c.add_argument("--workers", type=int, default=16)
    t = sub.add_parser("train")
    t.add_argument("--cache", type=Path, nargs="+", required=True); t.add_argument("--val", type=Path)
    t.add_argument("--out", type=Path, required=True); t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch-size", type=int, default=4); t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--f0", type=int, default=32); t.add_argument("--refine-from", type=int, default=2)
    t.add_argument("--workers", type=int, default=8); t.add_argument("--init", type=Path)
    t.add_argument("--max-val", type=int, default=200)
    e = sub.add_parser("eval")
    e.add_argument("--data", type=Path, required=True); e.add_argument("--split", default="test")
    e.add_argument("--model", type=Path, required=True); e.add_argument("--out", type=Path, required=True)
    e.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.cmd == "cache":
        print("cached", build_cache(a.data, a.split, a.out, a.workers))
    elif a.cmd == "train":
        a.out.parent.mkdir(parents=True, exist_ok=True)
        train(a.cache, a.val, a.out, a.epochs, a.batch_size, a.lr, a.f0, a.refine_from, a.workers, a.init, a.max_val)
    else:
        evaluate(a.data, a.split, a.model, a.out, a.limit)


if __name__ == "__main__":
    main()
