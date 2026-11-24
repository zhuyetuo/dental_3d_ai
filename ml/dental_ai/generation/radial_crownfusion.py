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


def evaluate(data: Path, model_path: Path | None, out: Path, steps: int, refine: float | None, keep_sigma: float | None, limit: int | None,
             save: int, root: Path, clearance: float | None = None, split: str = "test", blend: float | None = None,
             detail_gain: float = 1.0) -> dict:
    """测试集：D 自己 / 扩散模型补完的牙冠，和技师牙冠比（深度图逐像素 + 网格距离 + 细节）。"""
    import torch
    from scipy.spatial import cKDTree

    from .radial import map_to_mesh
    from .radial_diffusion import (build_condition, clean_depth, decode_tooth, encode_tooth, load_model, sample)

    device = "cuda"
    files = sorted((root / "radial" / f"cf_{split}").glob("*.npz"))[:limit]
    model = load_model(model_path, device) if model_path else None
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
            with torch.no_grad():
                x = sample(model, cond, cls, steps, g, ant=ant if clearance is not None else None, clearance=clearance,
                           init=d_enc if refine is not None else None, strength=refine or 1.0,
                           keep_sigma=keep_sigma, base=d_enc if getattr(model, "residual", False) else None)
            preds["diff"] = decode_tooth(x)
            if blend:  # 大尺度形状完全用 D 的，只叠扩散模型的细节（各自减去自己的低通）
                from .radial_diffusion import lowpass

                p, dm = preds["diff"], dmap > 0
                both = dm & (p > 0)
                mixed = lowpass(dmap, dm, blend) + detail_gain * (p - lowpass(p, p > 0, blend))
                preds["diff"] = torch.where(both, mixed, dmap)
                if clearance is not None:  # 叠完细节再保证一次离对颌的距离
                    lim = torch.where(ant > 0, ant - clearance, torch.full_like(ant, 1e9))
                    preds["diff"] = torch.where(preds["diff"] > 0, torch.minimum(preds["diff"], lim).clamp(min=0.31), preds["diff"])
        stats = {k: detail_stats(p, tooth) for k, p in preds.items()}
        fine = {k: detail_stats(p, tooth, sigma=2.0) for k, p in preds.items()}  # 沟、窝这一级（约 0.3 mm）
        for j, i in enumerate(ok):
            case = batch[i].stem[2:]
            gt_mesh = trimesh.load(data / split / case / "outer_crown.ply", process=False)
            b10 = np.asarray(trimesh.sample.sample_surface(gt_mesh, 10000, seed=1)[0]) @ R_CF.T - z[i]["origin"][0]
            row = {"case": case}
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
    keys = sorted({k for r in rows for k in r if k != "case"})
    s = {"n": len(rows), **{k: round(float(np.nanmedian([r[k] for r in rows])), 4) for k in keys}}
    s.update({f"mean_{k}": round(float(np.nanmean([r[k] for r in rows])), 4) for k in keys if k.endswith(("_cd", "_cd_l2_cf"))})
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
    e.add_argument("--clearance", type=float, help="采样时每一步把牙冠限制在离对颌至少这么远（mm）")
    a = ap.parse_args()
    if a.cmd == "cache":
        print("cached", cache(a.data, a.split, a.gen_model, a.workers, a.root))
    else:
        print(json.dumps(evaluate(a.data, a.model, a.out, a.steps, a.refine, a.keep_sigma, a.limit, a.save, a.root, a.clearance, a.split,
                                  a.blend, a.detail_gain),
                         ensure_ascii=False))


if __name__ == "__main__":
    main()
