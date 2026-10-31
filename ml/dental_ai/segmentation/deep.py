"""深度分割模型（DGCNN 风格的点云网络），GPU 上训练，CPU 也能跑（慢）。

和 CPU 基线（baseline.py）用同一套输入特征和 17 类输出，区别是网络会看每个点周围 k 个邻居的形状（EdgeConv），
基线的弱点——牙齿边界——主要靠这个改善。

    # 1. 预处理：每个扫描采样一批点，存特征和标签（直接读压缩包，不用解压）
    python -m dental_ai.segmentation.deep cache --zips ../data/drive/raw/Teeth3DS --split <training_lower.txt> <training_upper.txt> \
        --extra-dir ../data/poseidon3d/train --out ../data/deep_cache
    # 2. 训练（自动用 CUDA）
    python -m dental_ai.segmentation.deep train --cache ../data/deep_cache --epochs 60 --out ../models/segmentation/deep.pt
    # 3. 预测（输出格式与 evaluation --pred-dir 一致）
    python -m dental_ai.segmentation.deep predict --model ../models/segmentation/deep.pt --data-dir ../data/teeth3ds_test100 --out ../pred/deep

GPU 服务器上的一键脚本见 scripts/gpu/。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ..fdi import Jaw
from .base import Segmenter
from .baseline import FEATURES, N_CLASSES, _adjacency, class_to_fdi, fdi_to_class, features, groove_adjacency

CACHE_POINTS = 24000  # 每个扫描缓存的点数（训练时再从中随机取 n_points 个）


def _torch():
    import torch

    return torch


def build_model(in_dim: int = len(FEATURES), k: int = 16, n_classes: int = N_CLASSES):
    torch = _torch()
    nn = torch.nn

    def knn(x, k):  # x: (B, C, N) → (B, N, k) 近邻下标（按特征空间距离）
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        dist = -xx - inner - xx.transpose(2, 1)
        return dist.topk(k=k, dim=-1)[1]

    def edge_features(x, idx):  # (B, C, N), (B, N, k) → (B, 2C, N, k)
        b, c, n = x.shape
        base = torch.arange(b, device=x.device).view(-1, 1, 1) * n
        flat = (idx + base).view(-1)
        xt = x.transpose(2, 1).contiguous()
        neigh = xt.view(b * n, -1)[flat].view(b, n, -1, c)
        center = xt.view(b, n, 1, c).expand(-1, -1, neigh.shape[2], -1)
        return torch.cat([neigh - center, center], dim=3).permute(0, 3, 1, 2).contiguous()

    class EdgeConv(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.mlp = nn.Sequential(nn.Conv2d(2 * cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2))

        def forward(self, x, idx):
            return self.mlp(edge_features(x, idx)).max(dim=-1)[0]

    class DGCNNSeg(nn.Module):
        def __init__(self):
            super().__init__()
            self.k = k
            self.e1, self.e2, self.e3 = EdgeConv(in_dim, 64), EdgeConv(64, 64), EdgeConv(64, 128)
            self.glob = nn.Sequential(nn.Conv1d(256, 512, 1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
            self.head = nn.Sequential(
                nn.Conv1d(256 + 512, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
                nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
                nn.Conv1d(128, n_classes, 1),
            )

        def forward(self, x):  # x: (B, C, N)
            idx = knn(x[:, :3], self.k)  # 第一层按坐标找近邻
            x1 = self.e1(x, idx)
            x2 = self.e2(x1, knn(x1, self.k))
            x3 = self.e3(x2, knn(x2, self.k))
            local = torch.cat([x1, x2, x3], dim=1)
            g = self.glob(local).max(dim=-1, keepdim=True)[0].expand(-1, -1, local.shape[2])
            return self.head(torch.cat([local, g], dim=1))  # (B, n_classes, N)

    return DGCNNSeg()


# ---------------------------------------------------------------- 数据


def _cache_one(args):
    scan, out_dir, seed = args
    dest = Path(out_dir) / f"{scan.name}.npz"
    if dest.exists():
        return scan.name
    try:
        if hasattr(scan, "load"):
            mesh, labels = scan.load()
        else:
            from ..io import load_labels, load_mesh

            mesh, labels = load_mesh(scan.mesh_path), load_labels(scan.gt_path)[0]
        x = features(mesh, scan.jaw)
        y = fdi_to_class(labels, scan.jaw)
        rng = np.random.default_rng(seed)
        referenced = np.unique(mesh.faces)
        pick = rng.choice(referenced, min(CACHE_POINTS, len(referenced)), replace=False)
        np.savez_compressed(dest, x=x[pick].astype(np.float16), y=y[pick].astype(np.uint8))
        return scan.name
    except Exception as exc:  # 个别扫描读不了不影响整体
        print(f"跳过 {scan.name}: {exc}", flush=True)
        return None


def build_cache(scans, out_dir: Path, workers: int = 4) -> int:
    from concurrent.futures import ProcessPoolExecutor

    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(s, out_dir, i) for i, s in enumerate(scans)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        done = [n for n in pool.map(_cache_one, jobs, chunksize=2) if n]
    # 特征标准化参数（训练和推理共用）
    sample = np.concatenate([np.load(p)["x"][:2000].astype(np.float32) for p in sorted(out_dir.glob("*.npz"))[:300]])
    stats = {"mean": sample.mean(axis=0).tolist(), "std": (sample.std(axis=0) + 1e-6).tolist(), "features": FEATURES}
    (out_dir / "stats.json").write_text(json.dumps(stats))
    return len(done)


# ---------------------------------------------------------------- 训练


def train(cache: Path, out: Path, epochs: int = 60, n_points: int = 8000, batch_size: int = 8, lr: float = 1e-3,
          val_fraction: float = 0.05, device: str | None = None, max_scans: int | None = None) -> dict:
    torch = _torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted(cache.glob("*.npz"))[:max_scans]
    stats = json.loads((cache / "stats.json").read_text())
    mean, std = np.array(stats["mean"], np.float32), np.array(stats["std"], np.float32)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(files))
    n_val = max(1, int(len(files) * val_fraction))
    val_files, train_files = [files[i] for i in order[:n_val]], [files[i] for i in order[n_val:]]
    data = {f: np.load(f) for f in files}  # 缓存整体约 1 GB（1350 个扫描），直接放内存
    data = {f: (d["x"].astype(np.float32), d["y"].astype(np.int64)) for f, d in data.items()}

    # 类别权重：牙龈点多，按频率的平方根反比加权
    counts = np.bincount(np.concatenate([data[f][1] for f in train_files]), minlength=N_CLASSES).astype(np.float64)
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)) ** 0.5, dtype=torch.float32)
    weights = weights / weights.mean()

    def batch(fs, augment):
        xs, ys = [], []
        for f in fs:
            x, y = data[f]
            idx = rng.choice(len(x), min(n_points, len(x)), replace=False)
            xb = (x[idx] - mean) / std
            if augment:  # 轻微缩放和抖动（坐标、尺度相关的特征一起缩放）
                xb = xb * rng.uniform(0.95, 1.05) + rng.normal(0, 0.01, xb.shape).astype(np.float32)
            xs.append(xb.T)
            ys.append(y[idx])
        return torch.tensor(np.stack(xs), device=device), torch.tensor(np.stack(ys), device=device)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, len(train_files) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    history, best = [], -1.0
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(len(train_files))
        total = 0.0
        for s in range(steps):
            xb, yb = batch([train_files[i] for i in perm[s * batch_size:(s + 1) * batch_size]], augment=True)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss)
        model.eval()
        with torch.no_grad():
            correct = n = 0
            for i in range(0, len(val_files), batch_size):
                xb, yb = batch(val_files[i:i + batch_size], augment=False)
                pred = model(xb).argmax(dim=1)
                correct += int((pred == yb).sum())
                n += yb.numel()
        acc = correct / max(n, 1)
        history.append({"epoch": epoch + 1, "loss": round(total / steps, 4), "val_point_acc": round(acc, 4), "minutes": round((time.time() - t0) / 60, 1)})
        print(json.dumps(history[-1]), flush=True)
        if acc > best:
            best = acc
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "stats": stats, "n_points": n_points, "history": history}, out)
    return {"best_val_point_acc": best, "epochs": epochs, "train_scans": len(train_files), "device": device}


# ---------------------------------------------------------------- 推理


class DeepSegmenter(Segmenter):
    name = "deep_dgcnn"
    postprocess = True

    def __init__(self, model_path: str | Path, device: str | None = None, passes: int = 3, refine_iters: int = 20):
        torch = _torch()
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = build_model().to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.mean = np.array(ckpt["stats"]["mean"], np.float32)
        self.std = np.array(ckpt["stats"]["std"], np.float32)
        self.n_points = ckpt["n_points"]
        self.passes = passes
        self.refine_iters = refine_iters

    def predict_proba(self, mesh: trimesh.Trimesh, jaw: Jaw) -> np.ndarray:
        x = (features(mesh, jaw) - self.mean) / self.std
        rng = np.random.default_rng(0)
        referenced = np.unique(mesh.faces)
        sums = np.zeros((len(mesh.vertices), N_CLASSES))
        hits = np.zeros(len(mesh.vertices))
        with self.torch.no_grad():
            for _ in range(self.passes):  # 多次随机采样取平均
                idx = rng.choice(referenced, min(self.n_points, len(referenced)), replace=False)
                logits = self.model(self.torch.tensor(x[idx].T[None], device=self.device))[0]
                prob = self.torch.softmax(logits, dim=0).T.cpu().numpy()
                # 每个顶点取最近 3 个采样点的概率平均
                d, nn = cKDTree(mesh.vertices[idx]).query(mesh.vertices, k=3)
                w = 1.0 / np.maximum(d, 1e-6)
                sums += (prob[nn] * w[..., None]).sum(axis=1) / w.sum(axis=1, keepdims=True)
                hits += 1
        proba = sums / hits[:, None]
        avg = _adjacency(mesh)
        for _ in range(3):
            proba = 0.5 * proba + 0.5 * (avg @ proba)
        if self.refine_iters:
            groove = groove_adjacency(mesh)
            for _ in range(self.refine_iters):
                proba = 0.3 * proba + 0.7 * (groove @ proba)
        return proba

    def predict(self, mesh: trimesh.Trimesh, jaw: Jaw) -> np.ndarray:
        return class_to_fdi(np.argmax(self.predict_proba(mesh, jaw), axis=1), jaw)


def main() -> None:
    from ..datasets import teeth3ds_zip_scans
    from ..evaluation import find_scans
    from ..io import load_mesh, save_labels

    p = argparse.ArgumentParser(description="深度分割模型（DGCNN）")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache", help="预处理训练数据")
    c.add_argument("--zips", type=Path, help="Teeth3DS 压缩包目录")
    c.add_argument("--split", type=Path, nargs="*", default=[])
    c.add_argument("--extra-dir", type=Path, nargs="*", default=[])
    c.add_argument("--limit", type=int)
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--n-points", type=int, default=8000)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--device")
    t.add_argument("--max-scans", type=int)
    t.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("predict")
    q.add_argument("--model", type=Path, required=True)
    q.add_argument("--data-dir", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--device")
    q.add_argument("--limit", type=int)
    args = p.parse_args()

    if args.cmd == "cache":
        scans = [s for f in args.split for s in teeth3ds_zip_scans(args.zips, f)] if args.zips else []
        for d in args.extra_dir:
            scans += find_scans(d, d)
        scans = scans[: args.limit]
        print(f"预处理 {len(scans)} 个扫描", flush=True)
        print(f"完成 {build_cache(scans, args.out, args.workers)} 个", flush=True)
    elif args.cmd == "train":
        info = train(args.cache, args.out, args.epochs, args.n_points, args.batch_size, args.lr, device=args.device, max_scans=args.max_scans)
        print(json.dumps(info, ensure_ascii=False))
    else:
        seg = DeepSegmenter(args.model, device=args.device)
        for i, scan in enumerate(find_scans(args.data_dir, args.data_dir)[: args.limit], 1):
            t0 = time.time()
            labels = seg.predict(load_mesh(scan.mesh_path), scan.jaw)
            dest = args.out / scan.case / f"{scan.name}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            save_labels(dest, labels, scan.jaw, patient_id=scan.case)
            print(f"[{i}] {scan.name} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
