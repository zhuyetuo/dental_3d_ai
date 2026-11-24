"""真实预备体数据（FDI 16 Crown Dataset）上的咬合面细节：径向深度图扩散模型在技师牙冠上训练。

    # 径向图缓存（技师牙冠 / 邻牙 / 对颌）+ 形状模型 D 的输出图，格式和 radial.py / radial_dmaps.py 一样
    python -m dental_ai.generation.radial_crownfusion cache --data ~/zyt/dental_data/crownfusion/CrownFusion --split train \\
        --gen-model ../models/generation/toothgen_cf16.pt
    # 训练：radial_diffusion train --data ../data/radial/cf_train --val ../data/radial/cf_val --with-d --residual ...
    # 评测：扩散模型在 D 上补细节，和技师牙冠比（外形 + 咬合面细节）
    python -m dental_ai.generation.radial_crownfusion eval --data ... --model <扩散模型> --out ../runs/crownfusion/detail

坐标和 eval_crownfusion 一样（标准 = 数据坐标 @ R.T）；原点 = 邻牙 15、17 质心的中点往牙龈方向降 CENTER_DROP（radial.gap_origin）。
细节指标：深度图减去自己的低通（σ = 6 像素）是“细节”；比生成的细节和技师牙冠细节的幅度比（1 最好）、相关系数（越高越好）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

R_CF = np.array([[0, 0, 1.0], [1.0, 0, 0], [0, 1.0, 0]])  # 行 = 标准坐标轴在数据坐标里的方向
_GEN = None


def _labels_for(dent: trimesh.Trimesh, pts, props) -> np.ndarray:
    """dentition_mesh 的每个顶点取最近输入点的类别：15 / 16（预备体）/ 17 / 对颌 40。"""
    from scipy.spatial import cKDTree

    cls = np.where(props[:, 1] == 1, 40, np.select([props[:, 0] == 1, props[:, 0] == -1], [15, 17], 16))
    return cls[cKDTree(pts[:, :3]).query(np.asarray(dent.vertices))[1]]


def _sample(verts, faces, n, seed):
    if len(faces) == 0:
        return np.zeros((0, 3))
    return np.asarray(trimesh.sample.sample_surface(trimesh.Trimesh(verts, faces, process=False), n, seed=seed)[0])


def case_maps(d: Path, gen_model=None, device="cpu") -> dict | None:
    from ..eval_crownfusion import load_case, run_case
    from .radial import gap_origin, splat
    from .radial_diffusion import clean_depth

    pts, props, prep, crown = load_case(d)
    dent = trimesh.load(d / "dentition_mesh.ply", process=False)
    lab = _labels_for(dent, pts, props)
    v = np.asarray(dent.vertices) @ R_CF.T
    f = np.asarray(dent.faces)
    fl = lab[f]
    xyz = pts[:, :3].astype(float) @ R_CF.T
    cents = {}
    for side, fdi in ((1, 15), (-1, 17)):
        sel = (props[:, 1] == 0) & (props[:, 0] == side)
        if sel.sum() >= 200:
            cents[fdi] = xyz[sel].mean(axis=0)
    origin = gap_origin(cents, 16, extrapolate=False)
    if origin is None:
        return None
    seed = int(d.name)
    zero = np.zeros(3)
    tooth = splat(_sample(np.asarray(crown.vertices) @ R_CF.T, np.asarray(crown.faces), 150000, seed) - origin, zero)
    ctx = splat(_sample(v, f[np.isin(fl, [15, 17]).all(axis=1)], 300000, seed + 1) - origin, zero)
    a = _sample(v, f[(fl == 40).all(axis=1)], 300000, seed + 2) - origin
    a = a[a[:, 2] > -1.0]
    ant = splat(a, zero) if len(a) > 1000 else np.zeros_like(tooth)
    out = {"fdi": 16, "origin": origin, "tooth": tooth, "ctx": ctx, "ant": ant, "has_ant": bool((ant > 0).sum() > 500)}
    if gen_model is not None:
        r = run_case(gen_model, d, 1.0, True, True, device)
        dm = np.zeros_like(tooth)
        if r is not None:
            g = _sample(np.asarray(r[1].vertices) @ R_CF.T, np.asarray(r[1].faces), 150000, 0)
            dm = clean_depth(splat(g - origin, zero))
        out["d"], out["has_d"] = dm, bool((dm > 0).sum() >= 1500)
    return out


def _init(gen_path):
    global _GEN
    import torch

    torch.set_num_threads(1)
    if gen_path:
        from .learned import load_model

        _GEN = load_model(Path(gen_path), "cuda")[0].eval()


def _cache_one(args):
    d, out, out_d = args
    dest, dest_d = out / f"cf{d.name}.npz", out_d / f"cf{d.name}.npz"
    if dest.exists() and dest_d.exists():
        return 1
    try:
        m = case_maps(d, _GEN, "cuda")
    except Exception as exc:  # noqa: BLE001
        print("skip", d.name, type(exc).__name__, exc, flush=True)
        return 0
    if m is None or (m["tooth"] > 0).sum() < 2000:
        return 0
    one = lambda k: np.asarray(m[k])[None]  # noqa: E731
    np.savez_compressed(dest, fdi=np.array([16], np.int16), origin=one("origin").astype(np.float32), has_ant=np.array([m["has_ant"]]),
                        **{k: one(k).astype(np.float16) for k in ("tooth", "ctx", "ant")})
    np.savez_compressed(dest_d, fdi=np.array([16], np.int16), d=one("d").astype(np.float16), has_d=np.array([m["has_d"]]))
    return 1


SPLITS = {"test": "testing.json", "val": "validation.json", "train": "training.json"}


def cache(data: Path, split: str, gen_model: str, workers: int, root: Path) -> int:
    import multiprocessing as mp

    ids = json.load(open(data / SPLITS[split]))
    out, out_d = root / "radial" / f"cf_{split}", root / "radial_d" / f"cf_{split}"
    out.mkdir(parents=True, exist_ok=True), out_d.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(gen_model,)) as pool:
        return sum(pool.imap_unordered(_cache_one, [(data / split / str(c), out, out_d) for c in ids], chunksize=4))


def detail_stats(pred, gt, sigma=6.0):
    """细节（深度图减去自己的低通）在两边都有牙冠、离边缘 4 像素以内不算的地方：幅度比、相关系数。pred、gt (B, S, S) mm。"""
    import torch
    import torch.nn.functional as F

    from .radial_diffusion import lowpass

    m = (pred > 0) & (gt > 0)
    m = -F.max_pool2d(-m.float()[:, None], 9, 1, 4)[:, 0] > 0.5
    hp = pred - lowpass(pred, pred > 0, sigma)
    hg = gt - lowpass(gt, gt > 0, sigma)
    out = []
    for i in range(len(pred)):
        a, b = hp[i][m[i]], hg[i][m[i]]
        if len(a) < 500:
            out.append((np.nan, np.nan))
            continue
        out.append((float(a.std() / b.std().clamp(min=1e-6)), float(torch.corrcoef(torch.stack([a, b]))[0, 1])))
    return out


def build_library(data: Path, out: Path, workers: int = 32) -> dict:
    """训练集的技师牙冠 → 细节库（generation/detail.py 的格式，标准坐标、以质心为原点，来源 2 = 技师牙冠）。"""
    from concurrent.futures import ProcessPoolExecutor

    from .detail import _save_class
    from .learned import target_class

    ids = json.load(open(data / SPLITS["train"]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        crowns = [c for c in pool.map(_lib_one, [data / "train" / str(i) for i in ids], chunksize=16) if c is not None]
    out.mkdir(parents=True, exist_ok=True)
    _save_class(out, target_class(16), crowns, [2] * len(crowns))
    info = {"technician_crowns": len(crowns), "of": len(ids)}
    (out / "library.json").write_text(json.dumps(info))
    return info


def _lib_one(d: Path):
    from .detail import MIN_VERTS, n_boundaries

    try:
        m = trimesh.load(d / "outer_crown.ply", process=False)
        v, f = np.asarray(m.vertices, float) @ R_CF.T, np.asarray(m.faces, np.int32)
        if len(v) < MIN_VERTS or n_boundaries(f) != 1:
            return None
        return (v - v.mean(axis=0)).astype(np.float32), f
    except Exception:  # noqa: BLE001
        return None


def lib_pick(lib, pred: np.ndarray, d_map: np.ndarray, origin: np.ndarray, k: int, by: str = "detail", max_fit: float = 0.3):
    """细节库里取外形最像的 k 颗、各自贴到 D 上；by="detail" 时选“细节起伏和 D 的细节最一致”的那颗（D 光滑，但它隐约的起伏
    和技师牙冠沟窝的相关有 0.4，比随便一颗库牙准），by="fit" 时选贴合误差最小的（detail.DetailLibrary.refine 的做法）。
    返回 (深度图, 贴合误差 mm) 或 None。pred：D 的顶点（标准坐标）；d_map：D 的径向图。"""
    import torch

    from .detail import chamfer, fit, support
    from .learned import target_class
    from .radial import splat
    from .radial_diffusion import clean_depth

    cls = target_class(16)
    d = lib._class(cls)
    center = pred.mean(axis=0)
    target = pred - center
    from scipy.spatial import cKDTree

    tree = cKDTree(target)
    order = np.argsort(np.linalg.norm(d["desc"] - support(target), axis=1))[:k]
    dm = torch.tensor(d_map, dtype=torch.float32)[None]
    best = None
    for i in order:
        v, f = lib.crown(cls, int(i))
        x = fit(v, f, target, tree)
        tx = cKDTree(x)
        c = chamfer(x, target, ta=tx, tb=tree)
        if c > max_fit or np.percentile(tx.query(target)[0], 95) > 0.5:
            continue
        if by == "fit":
            score = -c
            depth = None
        else:
            depth = clean_depth(splat(_sample(x + center, f, 150000, 0) - origin, np.zeros(3)))
            st = detail_stats(torch.tensor(depth, dtype=torch.float32)[None], dm, 2.0)[0][1] + detail_stats(
                torch.tensor(depth, dtype=torch.float32)[None], dm, 6.0)[0][1]
            score = -np.inf if np.isnan(st) else st
        if best is None or score > best[0]:
            best = (score, x, f, depth, c)
    if best is None:
        return None
    _, x, f, depth, c = best
    if depth is None:
        depth = clean_depth(splat(_sample(x + center, f, 150000, 0) - origin, np.zeros(3)))
    return depth, c


def lib_pick_gpu(lib, pred: np.ndarray, d_map: np.ndarray, origin: np.ndarray, k: int, by: str = "detail", max_fit: float = 0.3,
                 device: str = "cuda", n_pts: int = 150000, chunk: int = 8):
    """lib_pick 的 GPU 批量版：k 颗候选的缩放、ICP、低频形变、贴合误差、表面采样、径向投影、细节相关一次算完。
    和 CPU 版的差别：投影的补洞用 3×3 十字闭运算近似 ndimage 的默认结构（同一个），采样的随机数不同。"""
    import torch
    import torch.nn.functional as F

    from .detail import support
    from .learned import target_class
    from .radial import MAX_POLAR, R_CLIP, SIZE
    from .radial_diffusion import clean_depth

    cls = target_class(16)
    d = lib._class(cls)
    center = pred.mean(axis=0)
    target = pred - center
    order = np.argsort(np.linalg.norm(d["desc"] - support(target), axis=1))[:k]
    vs, fs = zip(*[lib.crown(cls, int(i)) for i in order])
    K, Vm, Fm = len(vs), max(len(v) for v in vs), max(len(f) for f in fs)
    X = torch.zeros(K, Vm, 3, device=device)
    M = torch.zeros(K, Vm, dtype=torch.bool, device=device)
    Fp = torch.zeros(K, Fm, 3, dtype=torch.long, device=device)
    FM = torch.zeros(K, Fm, dtype=torch.bool, device=device)
    for j, (v, f) in enumerate(zip(vs, fs)):
        X[j, :len(v)] = torch.tensor(v, device=device, dtype=torch.float32)
        M[j, :len(v)] = True
        Fp[j, :len(f)] = torch.tensor(f.astype(np.int64), device=device)
        FM[j, :len(f)] = True
    T = torch.tensor(target, device=device, dtype=torch.float32)
    w = M.float()[..., None]
    big = 1e6

    def nn_to_target(A):  # (K,V,3) → 到目标最近点的下标、距离
        idx, dist = [], []
        for s in range(0, K, chunk):
            dd = torch.cdist(A[s:s + chunk], T[None].expand(min(chunk, K - s), -1, -1))
            m, i = dd.min(2)
            idx.append(i), dist.append(m)
        return torch.cat(idx), torch.cat(dist)

    def target_to(A):  # 目标每个点到候选最近点的距离 (K,M)
        out = []
        for s in range(0, K, chunk):
            a = torch.where(M[s:s + chunk, :, None], A[s:s + chunk], torch.full_like(A[s:s + chunk], big))
            out.append(torch.cdist(T[None].expand(len(a), -1, -1), a).min(2)[0])
        return torch.cat(out)

    def kabsch(A, B):
        n = w.sum(1, keepdim=True)
        ca, cb = (A * w).sum(1, keepdim=True) / n, (B * w).sum(1, keepdim=True) / n
        H = ((A - ca) * w).transpose(1, 2) @ (B - cb)
        U, _, Vh = torch.linalg.svd(H)
        V = Vh.transpose(1, 2)
        sgn = torch.sign(torch.det(V @ U.transpose(1, 2)))
        D = torch.diag_embed(torch.stack([torch.ones_like(sgn), torch.ones_like(sgn), sgn], 1))
        R = V @ D @ U.transpose(1, 2)
        return R, cb - ca @ R.transpose(1, 2)

    lo = torch.where(M[..., None], X, torch.full_like(X, big)).amin(1)
    hi = torch.where(M[..., None], X, torch.full_like(X, -big)).amax(1)
    scale = ((T.amax(0) - T.amin(0)) / (hi - lo)).clamp(0.8, 1.25)
    X = X * scale[:, None]
    X0 = X.clone()
    for _ in range(10):
        i, _ = nn_to_target(X)
        R, t = kabsch(X, T[i])
        X = X @ R.transpose(1, 2) + t
    R, _ = kabsch(X0, X)
    ang = torch.rad2deg(torch.arccos(((R.diagonal(dim1=1, dim2=2).sum(1) - 1) / 2).clamp(-1, 1)))
    shift = T.mean(0) - (X0 * w).sum(1) / w.sum(1)
    X = torch.where((ang > 35)[:, None, None], X0 + shift[:, None], X)
    # 低频形变：位移场在网格上做 60 次均匀平均（和 detail._smooth 一样：邻接 + 自己，按行归一）
    i, _ = nn_to_target(X)
    disp = (T[i] - X).reshape(-1, 3)
    base = (torch.arange(K, device=device) * Vm)[:, None, None]
    g = (Fp + base)[FM]  # (总三角形数, 3) 全局顶点号
    e = torch.cat([g[:, [0, 1]], g[:, [1, 2]], g[:, [2, 0]]])
    eye = torch.arange(K * Vm, device=device)
    rows = torch.cat([e[:, 0], e[:, 1], eye])
    cols = torch.cat([e[:, 1], e[:, 0], eye])
    A = torch.sparse_coo_tensor(torch.stack([rows, cols]), torch.ones(len(rows), device=device), (K * Vm, K * Vm)).coalesce()
    deg = torch.sparse.sum(A, 1).to_dense()
    A = torch.sparse_coo_tensor(A.indices(), A.values() / deg[A.indices()[0]], A.shape).coalesce()
    for _ in range(60):
        disp = torch.sparse.mm(A, disp)
    X = X + disp.reshape(K, Vm, 3)
    _, d_xt = nn_to_target(X)
    d_tx = target_to(X)
    c = 0.5 * ((d_xt * M).sum(1) / M.sum(1) + d_tx.mean(1))
    miss = torch.quantile(d_tx, 0.95, dim=1)
    ok = (c <= max_fit) & (miss <= 0.5)
    if not ok.any():
        return None
    # 表面按面积采样 → 径向投影（和 radial.splat 一样：每个像素取最近的表面，再补被包围的小洞）
    tri = X[torch.arange(K, device=device)[:, None, None], Fp]  # (K,Fm,3,3)
    area = 0.5 * torch.linalg.norm(torch.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0], dim=-1), dim=-1) * FM
    pick = torch.multinomial(area, n_pts, replacement=True)
    tp = torch.gather(tri, 1, pick[..., None, None].expand(-1, -1, 3, 3))
    r1, r2 = torch.rand(K, n_pts, 1, device=device).sqrt(), torch.rand(K, n_pts, 1, device=device)
    P = (1 - r1) * tp[:, :, 0] + r1 * (1 - r2) * tp[:, :, 1] + r1 * r2 * tp[:, :, 2]
    P = P + torch.tensor(center - origin, device=device, dtype=torch.float32)
    r = torch.linalg.norm(P, dim=-1)
    theta = torch.arccos((P[..., 2] / r.clamp(min=1e-9)).clamp(-1, 1))
    phi = torch.atan2(P[..., 1], P[..., 0])
    rho = theta / MAX_POLAR
    keep = (r > 1e-6) & (r < R_CLIP) & (rho < 1)
    ii = ((rho * torch.cos(phi) * 0.5 + 0.5) * SIZE).floor().long().clamp(0, SIZE - 1)
    jj = ((rho * torch.sin(phi) * 0.5 + 0.5) * SIZE).floor().long().clamp(0, SIZE - 1)
    lin = (torch.arange(K, device=device)[:, None] * SIZE * SIZE + ii * SIZE + jj)[keep]
    out = torch.full((K * SIZE * SIZE,), float("inf"), device=device).scatter_reduce(0, lin, r[keep], "amin").view(K, 1, SIZE, SIZE)
    valid = torch.isfinite(out)

    def dil(m):  # 3×3 十字膨胀（ndimage 默认结构）
        p = F.pad(m.float(), (1, 1, 1, 1))
        return (p[..., 1:-1, 1:-1] + p[..., :-2, 1:-1] + p[..., 2:, 1:-1] + p[..., 1:-1, :-2] + p[..., 1:-1, 2:]) > 0

    m = valid
    for _ in range(2):
        m = dil(m)
    for _ in range(2):
        m = ~dil(~m)
    holes = m & ~valid
    for _ in range(2):
        near = -F.max_pool2d(-torch.where(torch.isfinite(out), out, torch.full_like(out, 1e9)), 3, 1, 1)
        near = torch.where(near >= 1e9, torch.full_like(near, float("inf")), near)
        out = torch.where(holes & ~torch.isfinite(out), near, out)
    depth = torch.where(torch.isfinite(out), out, torch.zeros_like(out))[:, 0]
    if by == "fit":
        score = torch.where(ok, -c, torch.full_like(c, -big))
    else:
        dm = torch.tensor(d_map, device=device, dtype=torch.float32)[None].expand(K, -1, -1)
        s2 = torch.tensor([x[1] for x in detail_stats(depth, dm, 2.0)], device=device)
        s6 = torch.tensor([x[1] for x in detail_stats(depth, dm, 6.0)], device=device)
        score = torch.nan_to_num(s2 + s6, nan=-big)
        score = torch.where(ok, score, torch.full_like(score, -big))
    b = int(score.argmax())
    return clean_depth(depth[b].cpu().numpy()), float(c[b])


def _blend(dmap, p, sigma, gain, ant, clearance):
    """大尺度用 D 的、细节用 p 的（各自减去低通）；再保证离对颌的距离。"""
    import torch

    from .radial_diffusion import lowpass

    dm = dmap > 0
    both = dm & (p > 0)
    out = torch.where(both, lowpass(dmap, dm, sigma) + gain * (p - lowpass(p, p > 0, sigma)), dmap)
    if clearance is not None:
        lim = torch.where(ant > 0, ant - clearance, torch.full_like(ant, 1e9))
        out = torch.where(out > 0, torch.minimum(out, lim).clamp(min=0.31), out)
    return out


def evaluate(data: Path, model_path: Path | None, out: Path, steps: int, refine: float | None, keep_sigma: float | None, limit: int | None,
             save: int, root: Path, clearance: float | None = None, split: str = "test", blend: float | None = None,
             detail_gain: float = 1.0, n_samples: int = 1, library: Path | None = None, gen_model: str | None = None,
             lib_blend: float | None = None, lib_candidates: int = 16, lib_max_fit: float = 0.25, lib_by: str = "detail",
             shard: str | None = None, lib_gpu: bool = False) -> dict:
    """测试集：D 自己 / 扩散模型补完的牙冠，和技师牙冠比（深度图逐像素 + 网格距离 + 细节）。"""
    import torch
    from scipy.spatial import cKDTree

    from .radial import map_to_mesh
    from .radial_diffusion import (build_condition, clean_depth, decode_tooth, encode_tooth, load_model, sample)

    device = "cuda"
    files = sorted((root / "radial" / f"cf_{split}").glob("*.npz"))[:limit]
    if shard:  # "i/n"：只跑第 i 份（并行用，最后 merge）
        i_sh, n_sh = map(int, shard.split("/"))
        files = files[i_sh::n_sh]
    model = load_model(model_path, device) if model_path else None
    lib = gen = None
    if library is not None:
        from ..eval_crownfusion import run_case
        from .detail import DetailLibrary
        from .learned import load_model as load_gen

        lib = DetailLibrary(library, candidates=lib_candidates, max_fit_mm=lib_max_fit, max_miss_mm=0.5)
        gen = load_gen(Path(gen_model), device)[0].eval()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k0 in range(0, len(files), 32):
        batch = files[k0:k0 + 32]
        z = [np.load(f) for f in batch]
        zd = [np.load(root / "radial_d" / f"cf_{split}" / f.name) for f in batch]
        ok = [i for i in range(len(batch)) if zd[i]["has_d"][0]]
        if not ok:
            continue
        T = lambda key, src: torch.tensor(np.stack([src[i][key][0] for i in ok]).astype(np.float32), device=device)  # noqa: E731
        tooth, ctx, ant, dmap = T("tooth", z), T("ctx", z), T("ant", z), T("d", zd)
        has = torch.tensor([float(z[i]["has_ant"][0]) for i in ok], device=device)
        preds = {"D": dmap}
        if model is not None:
            cond = build_condition(ctx, ant, has)
            d_enc = encode_tooth(dmap)
            if model.cond_ch > 5:
                cond = torch.cat([cond, d_enc], dim=1)
            g = torch.Generator(device=device).manual_seed(0)
            from .learned import target_class

            cls = torch.full((len(ok),), target_class(16), device=device)
            outs = []
            for _ in range(n_samples):  # 几次采样取平均：接近“最可能的沟在哪”，位置更准、幅度变小（再用 detail_gain 放大）
                with torch.no_grad():
                    x = sample(model, cond, cls, steps, g, ant=ant if clearance is not None else None, clearance=clearance,
                               init=d_enc if refine is not None else None, strength=refine or 1.0,
                               keep_sigma=keep_sigma, base=d_enc if getattr(model, "residual", False) else None)
                outs.append(decode_tooth(x))
            st = torch.stack(outs)
            preds["diff"] = torch.where((st > 0).all(0), st.mean(0), outs[0])
            if blend:  # 大尺度形状完全用 D 的，只叠扩散模型的细节（各自减去自己的低通）
                from .radial_diffusion import lowpass

                p, dm = preds["diff"], dmap > 0
                both = dm & (p > 0)
                mixed = lowpass(dmap, dm, blend) + detail_gain * (p - lowpass(p, p > 0, blend))
                preds["diff"] = torch.where(both, mixed, dmap)
                if clearance is not None:  # 叠完细节再保证一次离对颌的距离
                    lim = torch.where(ant > 0, ant - clearance, torch.full_like(ant, 1e9))
                    preds["diff"] = torch.where(preds["diff"] > 0, torch.minimum(preds["diff"], lim).clamp(min=0.31), preds["diff"])
        lib_used = {}
        if lib is not None:  # 细节库：D 的牙冠在技师牙冠库里找最像的、贴上去（低频跟 D、细节来自技师牙冠）
            from .learned import target_class
            from .radial import splat

            maps = []
            for j, i in enumerate(ok):
                case = batch[i].stem[2:]
                r = run_case(gen, data / split / case, 1.0, True, True, device)
                res = (lib_pick_gpu if lib_gpu else lib_pick)(lib, np.asarray(r[1].vertices) @ R_CF.T, dmap[j].cpu().numpy(), z[i]["origin"][0], lib_candidates,
                               lib_by, lib_max_fit) if r is not None else None
                if res is None:
                    maps.append(dmap[j].cpu().numpy())
                    lib_used[case] = None
                    continue
                maps.append(res[0])
                lib_used[case] = res[1]
            preds["lib"] = torch.tensor(np.stack(maps).astype(np.float32), device=device)
            if lib_blend:
                preds["lib"] = _blend(dmap, preds["lib"], lib_blend, 1.0, ant, clearance)
            elif clearance is not None:
                lim = torch.where(ant > 0, ant - clearance, torch.full_like(ant, 1e9))
                preds["lib"] = torch.where(preds["lib"] > 0, torch.minimum(preds["lib"], lim).clamp(min=0.31), preds["lib"])
        stats = {k: detail_stats(p, tooth) for k, p in preds.items()}
        fine = {k: detail_stats(p, tooth, sigma=2.0) for k, p in preds.items()}  # 沟、窝这一级（约 0.3 mm）
        for j, i in enumerate(ok):
            case = batch[i].stem[2:]
            gt_mesh = trimesh.load(data / split / case / "outer_crown.ply", process=False)
            b10 = np.asarray(trimesh.sample.sample_surface(gt_mesh, 10000, seed=1)[0]) @ R_CF.T - z[i]["origin"][0]
            row = {"case": case}
            if lib is not None:
                row["lib_used"] = float(lib_used.get(case) is not None)
            for name, p in preds.items():
                depth = clean_depth(p[j].cpu().numpy())
                mesh = map_to_mesh(depth, np.zeros(3))
                a10 = np.asarray(trimesh.sample.sample_surface(mesh, 10000, seed=1)[0])
                d_ab, d_ba = cKDTree(b10).query(a10)[0], cKDTree(a10).query(b10)[0]
                gm = (tooth[j] > 0) & (p[j] > 0)
                am = (p[j] > 0) & (ant[j] > 0)  # 同一方向上牙冠比对颌还远 = 咬进对颌
                row[f"{name}_pen_mm"] = float((p[j] - ant[j]).clamp(min=0)[am].max()) if am.any() else 0.0
                if name == "D":
                    gam = (tooth[j] > 0) & (ant[j] > 0)
                    row["real_pen_mm"] = float((tooth[j] - ant[j]).clamp(min=0)[gam].max()) if gam.any() else 0.0
                row.update({f"{name}_cd": float((d_ab.mean() + d_ba.mean()) / 2),
                            f"{name}_cd_l2_cf": float(np.sqrt((d_ab ** 2).mean() + (d_ba ** 2).mean())),
                            f"{name}_depth_mae": float((p[j] - tooth[j]).abs()[gm].mean()),
                            f"{name}_detail_ratio": stats[name][j][0], f"{name}_detail_corr": stats[name][j][1],
                            f"{name}_fine_ratio": fine[name][j][0], f"{name}_fine_corr": fine[name][j][1]})
                if len(rows) < save:
                    (mesh.copy().apply_transform(np.eye(4))).export(out / f"{case}_{name}.ply")
            if len(rows) < save:
                np.save(out / f"{case}_origin.npy", z[i]["origin"][0])
            rows.append(row)
    (out / "rows.json").write_text(json.dumps(rows))
    return summarize_rows(rows, out)


def summarize_rows(rows: list[dict], out: Path) -> dict:
    keys = sorted({k for r in rows for k in r if k != "case"})
    s = {"n": len(rows), **{k: round(float(np.nanmedian([r[k] for r in rows])), 4) for k in keys}}
    s.update({f"mean_{k}": round(float(np.nanmean([r[k] for r in rows])), 4) for k in keys if k.endswith(("_cd", "_cd_l2_cf"))})
    if rows and "lib_used" in rows[0]:
        s["lib_used_rate"] = round(float(np.mean([r["lib_used"] for r in rows])), 4)
    s.update({f"rate_{k}": round(float(np.mean([r[k] > 0.1 for r in rows])), 4) for k in keys if k.endswith("pen_mm")})  # 咬进 > 0.1 mm 的比例
    (out / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1))
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--data", type=Path, required=True)
    c.add_argument("--split", required=True)
    c.add_argument("--gen-model", required=True)
    c.add_argument("--workers", type=int, default=16)
    c.add_argument("--root", type=Path, default=Path("../data"))
    e = sub.add_parser("eval")
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--model", type=Path)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--steps", type=int, default=50)
    e.add_argument("--refine", type=float)
    e.add_argument("--keep-sigma", type=float)
    e.add_argument("--limit", type=int)
    e.add_argument("--save", type=int, default=12)
    e.add_argument("--root", type=Path, default=Path("../data"))
    e.add_argument("--split", default="test", help="调参用 val，最后的数字用 test")
    e.add_argument("--blend", type=float, help="大尺度用 D、细节用扩散模型的分界（低通 σ，像素）")
    e.add_argument("--detail-gain", type=float, default=1.0, help="和 --blend 一起：细节乘这个倍数")
    e.add_argument("--n-samples", type=int, default=1, help="每颗牙采样几次取平均")
    e.add_argument("--library", type=Path, help="技师牙冠细节库（library 子命令生成）")
    e.add_argument("--gen-model", default="../models/generation/toothgen_cf16.pt", help="细节库要重新跑形状模型拿到网格")
    e.add_argument("--lib-blend", type=float, help="细节库也只取细节、大尺度用 D（低通 σ，像素）")
    e.add_argument("--lib-candidates", type=int, default=16)
    e.add_argument("--lib-max-fit", type=float, default=0.25, help="贴合误差超过这个（mm）就不用库牙")
    e.add_argument("--lib-by", default="detail", choices=["detail", "fit"], help="候选里按什么挑：细节和 D 一致 / 贴合误差最小")
    e.add_argument("--lib-gpu", action="store_true", help="细节库的候选贴合、投影在 GPU 上批量算（lib_pick_gpu）")
    e.add_argument("--shard", help="i/n：只跑第 i 份（几个进程并行，再用 merge 合并）")
    mg = sub.add_parser("merge", help="把几个 --shard 的输出目录合并成一份（rows.json + summary.json），生成的牙冠文件也拷过去")
    mg.add_argument("--out", type=Path, required=True)
    mg.add_argument("--parts", type=Path, nargs="+", required=True)
    lb = sub.add_parser("library")
    lb.add_argument("--data", type=Path, required=True)
    lb.add_argument("--out", type=Path, required=True)
    lb.add_argument("--workers", type=int, default=32)
    e.add_argument("--clearance", type=float, help="采样时每一步把牙冠限制在离对颌至少这么远（mm）")
    a = ap.parse_args()
    if a.cmd == "cache":
        print("cached", cache(a.data, a.split, a.gen_model, a.workers, a.root))
    elif a.cmd == "merge":
        import shutil

        a.out.mkdir(parents=True, exist_ok=True)
        rows = []
        for p in a.parts:
            rows += json.loads((p / "rows.json").read_text())
            for f in p.glob("*.ply"):
                shutil.copy(f, a.out / f.name)
            for f in p.glob("*_origin.npy"):
                shutil.copy(f, a.out / f.name)
        rows.sort(key=lambda r: r["case"])
        (a.out / "rows.json").write_text(json.dumps(rows))
        print(json.dumps(summarize_rows(rows, a.out), ensure_ascii=False))
    elif a.cmd == "library":
        print(json.dumps(build_library(a.data, a.out, a.workers)))
    else:
        print(json.dumps(evaluate(a.data, a.model, a.out, a.steps, a.refine, a.keep_sigma, a.limit, a.save, a.root, a.clearance, a.split,
                                  a.blend, a.detail_gain, a.n_samples, a.library, a.gen_model, a.lib_blend, a.lib_candidates,
                                  a.lib_max_fit, a.lib_by, a.shard, a.lib_gpu),
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
