"""学习式颈缘线检测（第一版）：点云网络给每个顶点预测“到颈缘线的有符号距离”，零等值线就是颈缘线。

和几何方法（detect.py）的区别：不需要分割给出的初始线，直接从预备体网格出发。
数据：teethPreparationData（52 个切牙代型，MIT，见 eval_margin.py），每个三角形标注颈缘线以上 / 以下。

    python -m dental_ai.margin.learned train --data-dir "../data/prep_margin/training Data" --out ../models/margin/margin.pt
    python -m dental_ai.margin.learned eval --model ../models/margin/margin.pt --data-dir "../data/prep_margin/test Data" --out ../runs/margin/eval_test

目标值：sign × min(到真值线的距离, CLIP) / CLIP，预备体一侧为正。特征：坐标、法向、凸棱强度（颈缘线是一圈凸棱）。
推理：把全部顶点随机分成若干组（每组 N_POINTS 个，和训练时的密度一致）分别预测，重复几轮取平均；
在网格上沿边线性插值取零点，连成闭合折线，取最长的一圈。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

N_POINTS = 12000
CLIP = 2.0  # mm
SCALE = 10.0


def case_arrays(path: Path):
    """→ 网格、特征 (V, 7)（坐标未居中）、目标 (V,)、真值线。"""
    from ..eval_margin import PREP, load_vtp, truth_line
    from .detect import edge_sharpness

    mesh, face_label = load_vtp(path)
    line = truth_line(mesh, face_label)
    v = np.asarray(mesh.vertices)
    prep = np.zeros(len(v), bool)
    np.logical_or.at(prep, mesh.faces.ravel(), np.repeat(face_label == PREP, 3))
    d = cKDTree(line).query(v)[0]
    target = np.where(prep, 1.0, -1.0) * np.minimum(d, CLIP) / CLIP
    feats = np.concatenate([v, mesh.vertex_normals, edge_sharpness(mesh)[:, None]], axis=1).astype(np.float32)
    return mesh, feats, target.astype(np.float32), line


def features_of(mesh: trimesh.Trimesh) -> np.ndarray:
    from .detect import edge_sharpness

    return np.concatenate([mesh.vertices, mesh.vertex_normals, edge_sharpness(mesh)[:, None]], axis=1).astype(np.float32)


def _prepare(feats: np.ndarray, rot: np.ndarray | None = None, scale: float = 1.0, shift: np.ndarray | None = None) -> np.ndarray:
    x = feats.copy()
    x[:, :3] -= x[:, :3].mean(axis=0)
    if rot is not None:
        x[:, :3], x[:, 3:6] = x[:, :3] @ rot.T, x[:, 3:6] @ rot.T
    x[:, :3] = x[:, :3] * scale / SCALE + (0 if shift is None else shift)
    return x


def _random_rotation(rng, flip: bool) -> np.ndarray:
    a = rng.uniform(0, 2 * np.pi)
    b, c = np.deg2rad(rng.uniform(-25, 25, 2))
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    ry = np.array([[np.cos(c), 0, np.sin(c)], [0, 1, 0], [-np.sin(c), 0, np.cos(c)]])
    m = rx @ ry @ rz
    return m @ np.diag([-1.0, 1.0, 1.0]) if flip else m  # 11 和 21 互为镜像


def _model():
    from ..segmentation.deep import build_model

    return build_model(in_dim=7, k=20, n_classes=1)


def predict_field(model, feats: np.ndarray, device: str, rounds: int = 3, seed: int = 0) -> np.ndarray:
    import torch

    rng = np.random.default_rng(seed)
    x = _prepare(feats)
    n = len(x)
    total, count = np.zeros(n), np.zeros(n)
    model.eval()
    with torch.no_grad():
        for _ in range(rounds):
            perm = rng.permutation(n)
            for s in range(0, n, N_POINTS):
                idx = perm[s:s + N_POINTS]
                if len(idx) < N_POINTS:
                    idx = np.concatenate([idx, rng.choice(n, N_POINTS - len(idx), replace=n < N_POINTS)])
                out = model(torch.tensor(x[idx].T[None]).to(device))[0, 0].cpu().numpy()
                np.add.at(total, idx, out)
                np.add.at(count, idx, 1)
    return total / np.maximum(count, 1)


def zero_contour(mesh: trimesh.Trimesh, field: np.ndarray, smooth_iters: int = 2) -> np.ndarray:
    """网格上标量场的零等值线：取最长的闭合折线，返回 (N, 3)。"""
    f = field.astype(float).copy()
    edges = mesh.edges_unique
    deg = np.bincount(edges.ravel(), minlength=len(f))
    for _ in range(smooth_iters):  # 轻度平滑，去掉零星的符号翻转
        s = np.zeros_like(f)
        np.add.at(s, edges[:, 0], f[edges[:, 1]])
        np.add.at(s, edges[:, 1], f[edges[:, 0]])
        f = 0.5 * f + 0.5 * s / np.maximum(deg, 1)
    f[f == 0] = 1e-9
    v = np.asarray(mesh.vertices)
    fe = mesh.faces_unique_edges  # (F, 3) 每个三角形的三条边
    cross = (f[edges[:, 0]] > 0) != (f[edges[:, 1]] > 0)
    t = f[edges[:, 0]] / (f[edges[:, 0]] - f[edges[:, 1]] + 1e-30)
    pts = v[edges[:, 0]] + t[:, None] * (v[edges[:, 1]] - v[edges[:, 0]])
    fc = cross[fe]
    link: dict[int, list[int]] = {}
    for tri in fe[fc.sum(axis=1) == 2]:
        a, b = tri[cross[tri]]
        link.setdefault(int(a), []).append(int(b))
        link.setdefault(int(b), []).append(int(a))
    seen, best = set(), []
    for start in link:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [e for e in link[cur] if e != prev and e not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            seen.add(cur)
            loop.append(cur)
        closed = len(loop) > 2 and loop[0] in link[loop[-1]]
        length = np.linalg.norm(np.diff(pts[loop], axis=0), axis=1).sum()
        if closed and length > (best[0] if best else 0):
            best = [length, loop]
    if not best:
        raise ValueError("没有找到闭合的零等值线")
    return pts[best[1]]


def detect(model, mesh: trimesh.Trimesh, device: str, feats: np.ndarray | None = None, n: int = 200):
    from .detect import resample_loop

    field = predict_field(model, features_of(mesh) if feats is None else feats, device)
    return resample_loop(zero_contour(mesh, field), n), field


# ---------------------------------------------------------------- 训练 / 评估


def train(data_dir: Path, out: Path, epochs: int, batch_size: int, lr: float, repeats: int, init: Path | None = None) -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(data_dir.glob("*.vtp"))
    cases = [case_arrays(f) for f in files]
    order = np.random.default_rng(0).permutation(len(cases))
    n_val = max(2, len(cases) // 8)
    val, tr = [cases[i] for i in order[:n_val]], [cases[i] for i in order[n_val:]]
    print(json.dumps({"train": len(tr), "val": n_val, "device": device}), flush=True)
    torch.manual_seed(0)
    model = _model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(tr) * repeats) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    best, t0 = float("inf"), time.time()

    def validate() -> float:
        errs = []
        for mesh, feats, _, line in val:
            try:
                pts, _ = detect(model, mesh, device, feats)
                errs.append(float(cKDTree(line).query(pts)[0].mean()))
            except ValueError:
                errs.append(5.0)
        return float(np.mean(errs))

    if init:  # 接着训练：先量一下起点，之后只有比它好才覆盖 out（out 可以就是 init）
        model.load_state_dict(torch.load(init, map_location=device)["state"])
        best = validate()
        print(json.dumps({"epoch": 0, "val_line_mm": round(best, 4), "note": "起点"}), flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        jobs =rng.permutation(np.repeat(np.arange(len(tr)), repeats))
        total, nb = 0.0, 0
        for s in range(0, len(jobs) - batch_size + 1, batch_size):
            xs, ys = [], []
            for j in jobs[s:s + batch_size]:
                _, feats, target, _ = tr[j]
                idx = rng.choice(len(feats), N_POINTS, replace=len(feats) < N_POINTS)
                x = _prepare(feats[idx], _random_rotation(rng, rng.random() < 0.5), rng.uniform(0.9, 1.1), rng.normal(0, 0.05, 3).astype(np.float32))
                xs.append(x.T), ys.append(target[idx])
            x = torch.tensor(np.stack(xs)).to(device)
            y = torch.tensor(np.stack(ys)).to(device)
            pred = model(x)[:, 0]
            w = 1 + 4 * torch.exp(-((y.abs() * CLIP / 0.5) ** 2))  # 颈缘线附近 0.5 mm 内权重更大
            loss = (w * torch.nn.functional.smooth_l1_loss(pred, y, beta=0.05, reduction="none")).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total, nb = total + loss.item(), nb + 1
        if epoch % 10 == 0 or epoch == epochs:
            v = validate()
            print(json.dumps({"epoch": epoch, "loss": round(total / nb, 4), "val_line_mm": round(v, 4), "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
            if v < best:
                best = v
                torch.save({"state": model.state_dict(), "epoch": epoch, "val_line_mm": v}, out)
    info = {"best_val_line_mm": best, "epochs": epochs}
    print(json.dumps(info), flush=True)
    return info


def run_eval(model_path: Path, data_dir: Path, out: Path) -> dict:
    import torch

    from ..eval_margin import PREP, line_errors
    from .detect import detect_margin, edge_sharpness

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=device)
    model = _model().to(device)
    model.load_state_dict(ckpt["state"])
    rows = []
    for f in sorted(data_dir.glob("*.vtp")):
        mesh, feats, target, line = case_arrays(f)
        tree = cKDTree(line)
        row = {"case": f.stem.replace("_labeledPrepFinal", "")}
        try:
            pts, field = detect(model, mesh, device, feats)
            row.update({f"learned_{k}": v for k, v in line_errors(pts, tree).items()})
            # 学习模型的结果当初始线，再用几何方法吸附到凸棱
            labels = np.where(field > 0, PREP, 0)
            up = (mesh.vertex_normals[field > 0]).sum(axis=0)
            up /= np.linalg.norm(up)
            res = detect_margin(mesh, labels, PREP, up=up, sharpness=edge_sharpness(mesh))
            row.update({f"snap_{k}": v for k, v in line_errors(res.points, tree).items()})
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    summary = {"cases": len(rows), "failed": sum("error" in r for r in rows), "model_epoch": ckpt["epoch"]}
    for prefix in ("learned", "snap"):
        ok = [r for r in rows if f"{prefix}_mean" in r]
        if ok:
            summary[prefix] = {k: round(float(np.median([r[f"{prefix}_{k}"] for r in ok])), 3) for k in ("mean", "p95", "max", "within_0.2")}
            summary[prefix]["mean_worst"] = round(max(r[f"{prefix}_mean"] for r in ok), 3)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "cases.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="学习式颈缘线检测")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data-dir", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=300)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--repeats", type=int, default=4, help="每轮每个预备体取几次随机子集")
    t.add_argument("--init", type=Path, help="从已有权重接着训练（可以和 --out 是同一个文件：只有验证集上更好才覆盖）；建议同时把 --lr 调小，如 3e-4")
    e = sub.add_parser("eval")
    e.add_argument("--model", type=Path, required=True)
    e.add_argument("--data-dir", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "train":
        train(args.data_dir, args.out, args.epochs, args.batch_size, args.lr, args.repeats, init=args.init)
    else:
        run_eval(args.model, args.data_dir, args.out)


if __name__ == "__main__":
    main()
