"""用 exocad 牙库造“设计风格”的训练样本：把同牙位的库牙对齐（旋转 + 平移 + 缩放）到每颗真牙上，
再用真牙的投影中心投成径向深度图。上下文、对颌不变，只是“答案”换成了技工设计用的标准牙形（牙尖、沟清楚）。
训练时这些样本用另一组牙位类别（cls + 16），推理时选这一组就是“设计风格”。

全在 GPU 上批量算：真牙点 (B, n, 3) 和库牙点 (B, m, 3) 两两最近点（cdist）、加权 Umeyama 一次解出 R、s、t。

    python -m dental_ai.generation.radial_library pilot --bank ../data/radial_lib/bank.npz --data ../data/radial/bite_val
    python -m dental_ai.generation.radial_library build --bank ../data/radial_lib/bank.npz --data ../data/radial/train ../data/radial/bite ../data/radial/bite_val
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .radial import SIZE, pixel_directions

N_FIT = 384     # 拟合时真牙取多少点
M_FIT = 1200    # 拟合时库牙取多少点
ITERS = 18
SCALE_RANGE = (0.65, 1.6)


def lib_dir_for(data_dir: Path) -> Path:
    """data/radial/<split> → data/radial_lib/<split>"""
    data_dir = Path(data_dir)
    return data_dir.parent.parent / "radial_lib" / data_dir.name


def map_points(depth: np.ndarray, n: int, rng) -> np.ndarray | None:
    """真牙的径向深度图 → 点（局部坐标，投影中心在原点），随机取 n 个。"""
    dirs, disk = pixel_directions()
    m = disk & (depth > 0.3)
    if m.sum() < 300:
        return None
    p = dirs[m] * depth[m][:, None]
    return p[rng.choice(len(p), n, replace=len(p) < n)].astype(np.float32)


def _rot_z(a):
    import torch

    c, s = torch.cos(a), torch.sin(a)
    z, o = torch.zeros_like(a), torch.ones_like(a)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


def init_rotations(yaws, zflip: bool, device):
    """库牙的初始朝向：先（可选）绕 x 转 180°（咬合方向反过来），再绕 z 转 yaw。"""
    import torch

    R = _rot_z(torch.as_tensor(yaws, dtype=torch.float32, device=device))
    if zflip:
        R = R @ torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device))
    return R


def umeyama(src, dst, w):
    """加权相似变换 dst ≈ s R src + t。src/dst (B, K, 3)，w (B, K)。"""
    import torch

    w = w / w.sum(1, keepdim=True).clamp(min=1e-8)
    ms, md = (w[..., None] * src).sum(1), (w[..., None] * dst).sum(1)
    xs, xd = src - ms[:, None], dst - md[:, None]
    cov = torch.einsum("bk,bki,bkj->bij", w, xd, xs)
    U, S, Vh = torch.linalg.svd(cov)
    d = torch.sign(torch.det(U @ Vh))
    D = torch.ones_like(S)
    D[:, 2] = d
    R = U @ torch.diag_embed(D) @ Vh
    var = (w * (xs ** 2).sum(-1)).sum(1).clamp(min=1e-8)
    s = ((S * D).sum(1) / var).clamp(*SCALE_RANGE)
    t = md - s[:, None] * torch.einsum("bij,bj->bi", R, ms)
    return R, s, t


def fit(P, L, R0, iters: int = ITERS, trim: float = 0.9):
    """P 真牙点 (B, n, 3)，L 库牙点 (B, m, 3)（库牙自己的坐标），R0 (B, 3, 3) 初始朝向。
    双向对应：真牙每个点 → 最近的库牙点；库牙“牙冠部分”（比真牙最低点高的）每个点 → 最近的真牙点。去掉最远的 10% 再解。
    返回 R、s、t、score（双向平均距离 mm）。"""
    import torch

    B = len(P)
    zmin = torch.quantile(P[..., 2], 0.03, dim=1)
    # 初始缩放：水平方向的展开度之比；初始平移：水平质心对齐，最高处对齐
    s = (P[..., :2].std(1).mean(1) / (L @ R0.transpose(1, 2))[..., :2].std(1).mean(1).clamp(min=1e-3)).clamp(*SCALE_RANGE)
    R = R0
    for it in range(iters + 1):
        Lr = s[:, None, None] * (L @ R.transpose(1, 2))
        if it == 0:
            t = torch.cat([P[..., :2].mean(1) - Lr[..., :2].mean(1),
                           (torch.quantile(P[..., 2], 0.97, dim=1) - torch.quantile(Lr[..., 2], 0.97, dim=1))[:, None]], 1)
        Lc = Lr + t[:, None]
        d = torch.cdist(P, Lc)                      # (B, n, m)
        d1, i1 = d.min(2)                           # 真牙 → 库牙
        crown = Lc[..., 2] > zmin[:, None] + 0.3    # 库牙在牙龈以上的部分
        d2, i2 = d.min(1)                           # 库牙 → 真牙（只算牙冠部分）
        d2 =torch.where(crown, d2, torch.zeros_like(d2))
        if it == iters:
            n2 = crown.sum(1).clamp(min=1)
            score = d1.mean(1) * 0.5 + (d2.sum(1) / n2) * 0.5
            return R, s, t, score, d1.mean(1)
        src = torch.cat([torch.gather(L, 1, i1[..., None].expand(-1, -1, 3)), L], 1)
        dst = torch.cat([P, torch.gather(P, 1, i2[..., None].expand(-1, -1, 3))], 1)
        dist = torch.cat([d1, d2], 1)
        w = torch.cat([torch.ones_like(d1), crown.float()], 1)
        thr = torch.quantile(torch.where(w > 0, dist, torch.full_like(dist, float("nan"))).nan_to_num(nan=1e3), trim, dim=1)
        w = w * (dist <= thr[:, None]).float()
        R, s, t = umeyama(src, dst, w)
    raise AssertionError


def load_bank(path: Path, device):
    import torch

    z = np.load(path)
    return {"pts": torch.tensor(z["pts"].astype(np.float32), device=device), "jaw": z["jaw"].astype(int), "pos": z["pos"].astype(int),
            "name": z["name"], "verts": torch.tensor(z["verts"], device=device), "faces": torch.tensor(z["faces"].astype(np.int64), device=device),
            "voff": z["voff"], "foff": z["foff"]}


def bank_mesh(bank, b: int):
    v = bank["verts"][bank["voff"][b]:bank["voff"][b + 1]]
    f = bank["faces"][bank["foff"][b]:bank["foff"][b + 1]]
    return v, f


def jaw_pos(fdi: int) -> tuple[int, int]:
    return (0 if fdi // 10 in (1, 2) else 1), fdi % 10


def _teeth(dirs, limit_jaws: int | None = None):
    rows = []
    for d in dirs:
        files = sorted(Path(d).glob("*.npz"))
        if limit_jaws:
            files = files[:limit_jaws]
        for f in files:
            with np.load(f) as z:
                for k, v in enumerate(z["fdi"]):
                    rows.append((f, k, int(v)))
    return rows


def pilot(bank_path, data_dirs, n_teeth: int, device: str = "cuda"):
    """看牙库的坐标约定：每个牙位用全套初始朝向（12 个 yaw × 咬合方向正反 × 左右镜像）试，统计哪种赢。"""
    import torch

    bank = load_bank(bank_path, device)
    rng = np.random.default_rng(0)
    rows = _teeth(data_dirs)
    rows = [rows[i] for i in rng.choice(len(rows), min(n_teeth, len(rows)), replace=False)]
    yaws = np.arange(12) * (2 * math.pi / 12)
    stats = {}
    for f, k, fdi in rows:
        if fdi % 10 == 8:
            continue
        with np.load(f) as z:
            P = map_points(z["tooth"][k].astype(np.float32), N_FIT, rng)
        jaw, pos = jaw_pos(fdi)
        cand = np.flatnonzero((bank["jaw"] == jaw) & (bank["pos"] == pos))
        if P is None or not len(cand):
            continue
        c = rng.choice(cand, 2, replace=False)
        best = None
        for mir in (1.0, -1.0):
            for zf in (False, True):
                R0 = init_rotations(yaws, zf, device)                  # (12, 3, 3)
                Lsub = bank["pts"][c][:, torch.randperm(bank["pts"].shape[1], device=device)[:M_FIT]]
                Lsub = Lsub * torch.tensor([mir, 1.0, 1.0], device=device)
                L = Lsub.repeat_interleave(12, 0)
                Pt = torch.tensor(P, device=device)[None].expand(len(L), -1, -1)
                _, s, _, score, _ = fit(Pt, L, R0.repeat(len(c), 1, 1))
                j = int(score.argmin())
                if best is None or float(score[j]) < best[0]:
                    best = (float(score[j]), mir, zf, int(j % 12), float(s[j]))
        key = f"{'upper' if jaw == 0 else 'lower'}{pos}"
        stats.setdefault(key, []).append(best)
    out = {}
    for key, v in sorted(stats.items()):
        a = np.array([[b[0], b[1], float(b[2]), b[3], b[4]] for b in v])
        modes = {}
        for b in v:
            modes[f"mir{int(b[1])}_zf{int(b[2])}"] = modes.get(f"mir{int(b[1])}_zf{int(b[2])}", 0) + 1
        out[key] = {"n": len(v), "score_mm": round(float(np.median(a[:, 0])), 3), "scale": round(float(np.median(a[:, 4])), 3),
                    "modes": modes, "yaw_hist": np.bincount(a[:, 3].astype(int), minlength=12).tolist()}
        print(key, json.dumps(out[key]), flush=True)
    return out


def build(bank_path, data_dirs, conv: dict, n_cand: int = 6, max_score: float = 0.35, min_cover: float = 0.92,
          device: str = "cuda", chunk: int = 384, seed: int = 0):
    """每颗真牙：同牙位随机 n_cand 颗库牙 × 3 个 yaw 初值（约定里的 yaw ± 30°），挑分最低的；
    分数 ≤ max_score 且覆盖真牙 ≥ min_cover 的才要。存 data/radial_lib/<split>/<同名>.npz：fdi、lib（深度图）、ok、score、name。"""
    import torch

    from .radial_diffusion import clean_depth
    from .radial_gpu import splat

    bank = load_bank(bank_path, device)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    _, disk = pixel_directions()
    for d in data_dirs:
        out_dir = lib_dir_for(Path(d))
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(Path(d).glob("*.npz"))
        jobs = []  # (file_idx, tooth_idx, fdi, P, cand)
        for fi, f in enumerate(files):
            with np.load(f) as z:
                for k, v in enumerate(z["fdi"]):
                    fdi = int(v)
                    jaw, pos = jaw_pos(fdi)
                    key = f"{'upper' if jaw == 0 else 'lower'}{pos}"
                    cand = np.flatnonzero((bank["jaw"] == jaw) & (bank["pos"] == pos))
                    if key not in conv or not len(cand):
                        continue
                    P = map_points(z["tooth"][k].astype(np.float32), N_FIT, rng)
                    if P is None:
                        continue
                    jobs.append((fi, k, fdi, P, rng.choice(cand, min(n_cand, len(cand)), replace=False)))
        res = {}
        # 拼成大批：每个任务 n_cand × 3 个问题
        probs = []
        for ji, (fi, k, fdi, P, cand) in enumerate(jobs):
            jaw, pos = jaw_pos(fdi)
            c = conv[f"{'upper' if jaw == 0 else 'lower'}{pos}"]
            for b in cand:
                for mir in c["mirs"]:
                    for y in c["yaws"]:
                        probs.append((ji, int(b), mir, False, y * (2 * math.pi / 12)))
        best = {}
        for s0 in range(0, len(probs), chunk):
            ch = probs[s0:s0 + chunk]
            P = torch.tensor(np.stack([jobs[p[0]][3] for p in ch]), device=device)
            idx = torch.randint(0, bank["pts"].shape[1], (len(ch), M_FIT), device=device)
            L = torch.gather(bank["pts"][[p[1] for p in ch]], 1, idx[..., None].expand(-1, -1, 3))
            mir = torch.tensor([[p[2], 1.0, 1.0] for p in ch], device=device)
            L = L * mir[:, None]
            R0 = torch.cat([init_rotations([p[4]], bool(p[3]), device) for p in ch])
            R, s, t, score, _ = fit(P, L, R0)
            for q, p in enumerate(ch):
                sc = float(score[q])
                if p[0] not in best or sc < best[p[0]][0]:
                    best[p[0]] = (sc, p[1], p[2], R[q], float(s[q]), t[q])
        # 投影：选中的库牙网格变换到局部坐标、表面采 15 万点（和真牙一样）→ 径向深度图，只留真牙轮廓（外扩 2 像素）以内
        from .radial_gpu import sample_surface

        g = torch.Generator(device=device).manual_seed(seed)
        by_file = {}
        for ji, (sc, b, mir, R, s, t) in best.items():
            by_file.setdefault(jobs[ji][0], []).append((ji, sc, b, mir, R, s, t))
        for fi, f in enumerate(files):
            with np.load(f) as z:
                T = len(z["fdi"])
                gt = z["tooth"].astype(np.float32)
                fdis = z["fdi"]
            lib = np.zeros((T, SIZE, SIZE), np.float16)
            ok = np.zeros(T, bool)
            score = np.full(T, np.nan, np.float32)
            names = np.array([""] * T, dtype=object)
            items = by_file.get(fi, [])
            if items:
                pts, bs = [], []
                for q, (ji, sc, b, mir, R, s, t) in enumerate(items):
                    v, fc = bank_mesh(bank, b)
                    Lb = sample_surface(v * torch.tensor([mir, 1.0, 1.0], device=device), fc, 150000, g)
                    pts.append(s * Lb @ R.T + t)
                    bs.append(torch.full((len(Lb),), q, device=device, dtype=torch.long))
                maps = splat(torch.cat(pts), torch.cat(bs), len(items)).cpu().numpy()
                for q, (ji, sc, b, mir, R, s, t) in enumerate(items):
                    k = jobs[ji][1]
                    m_gt = gt[k] > 0.3
                    grown = m_gt.copy()
                    from scipy import ndimage

                    grown = ndimage.binary_dilation(m_gt, iterations=2) & disk
                    dm = np.where(grown, maps[q], 0.0)
                    dm = clean_depth(dm)
                    cover = float(((dm > 0) & m_gt).sum() / max(1, m_gt.sum()))
                    score[k] = sc
                    names[k] = str(bank["name"][b])
                    if sc <= max_score and cover >= min_cover:
                        # 真牙轮廓里库牙没投到的少数像素用真牙补上，避免边上缺口
                        fill = m_gt & ~(dm > 0)
                        dm = np.where(fill, gt[k], dm)
                        lib[k] = dm.astype(np.float16)
                        ok[k] = True
            np.savez_compressed(out_dir / f.name, fdi=fdis, lib=lib, ok=ok, score=score, name=names.astype(str))
        n_ok = sum(int(np.load(out_dir / f.name)["ok"].sum()) for f in files)
        print(json.dumps({"split": str(d), "teeth": len(jobs), "ok": n_ok, "minutes": round((time.time() - t0) / 60, 1)}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="用 exocad 牙库造设计风格的训练样本")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pilot")
    a.add_argument("--bank", type=Path, required=True)
    a.add_argument("--data", type=Path, nargs="+", required=True)
    a.add_argument("--n", type=int, default=400)
    a.add_argument("--out", type=Path, default=None)
    b = sub.add_parser("build")
    b.add_argument("--bank", type=Path, required=True)
    b.add_argument("--data", type=Path, nargs="+", required=True)
    b.add_argument("--conv", type=Path, required=True, help="pilot 得出的约定（每个牙位的 mir、zf、yaw）")
    b.add_argument("--max-score", type=float, default=0.35)
    args = p.parse_args()
    if args.cmd == "pilot":
        out = pilot(args.bank, args.data, args.n)
        if args.out:
            args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        build(args.bank, args.data, conventions(json.loads(args.conv.read_text())), max_score=args.max_score)


def conventions(pilot_out: dict) -> dict:
    """pilot 的统计 → 每个牙位要试的镜像和 yaw：咬合方向都是 +z（pilot 里几乎全是 zf0）；
    镜像占比 ≥ 25% 的都试；yaw 取票数 ≥ 15% 的格子再各往两边扩一格。没统计到的牙位（第二磨牙票太少）借第一磨牙的。"""
    conv = {}
    for key, v in pilot_out.items():
        n = max(1, v["n"])
        mirs = {}
        for m, c in v["modes"].items():
            mir = float(m.split("_")[0][3:])
            mirs[mir] = mirs.get(mir, 0) + c
        hist = np.array(v["yaw_hist"])
        top = [i for i in range(12) if hist[i] >= 0.15 * n] or [int(hist.argmax())]
        yaws = sorted({(i + d) % 12 for i in top for d in (-1, 0, 1)})
        conv[key] = {"mirs": [m for m, c in mirs.items() if c >= 0.25 * n] or [1.0], "yaws": yaws}
    for jaw in ("upper", "lower"):
        for pos in range(1, 9):
            k = f"{jaw}{pos}"
            if k not in conv or pilot_out.get(k, {}).get("n", 0) < 8:
                conv[k] = conv.get(f"{jaw}6", conv.get(k))
    return {k: v for k, v in conv.items() if v}


if __name__ == "__main__":
    main()
