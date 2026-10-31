"""不需要 GPU 的分割基线：逐顶点几何特征 + 梯度提升树（scikit-learn），在 CPU 上训练和推理。

定位：TGNet 等深度模型需要 GPU，这里先有一个能在任何机器上跑、能用自己的数据训练的基线，
用来打通“数据 → 训练 → 官方指标评估 → 接入流程”的全链路，也作为深度模型的对照。

1. 朝向归一化（orientation.py）把扫描转到标准坐标系；
2. 每个顶点的特征：标准坐标、法向、两个尺度的曲率（相邻法向差）、相对局部最低点的高度、
   相对牙弓曲线的位置（沿牙弓的参数、到曲线的有符号距离、绕牙弓中心的角度）；
3. 17 类：0 牙龈，1–8 患者右侧第 1–8 颗，9–16 患者左侧第 1–8 颗（上下颌共用，加一个颌位特征）；
4. 推理后在网格上做几轮概率平滑，再映射成 FDI，交给编号纠错（numbering.py）。

    python -m dental_ai.segmentation.baseline train --zips ../data/drive/raw/Teeth3DS --split <训练集划分> --scans 300 --out ../models/segmentation/baseline.joblib
    python -m dental_ai.segmentation.baseline predict --model ../models/segmentation/baseline.joblib --data-dir ../data/teeth3ds_test100 --out ../pred/baseline
    python -m dental_ai.evaluation --data-dir ../data/teeth3ds_test100 --gt-dir ../data/teeth3ds_test100 --pred-dir ../pred/baseline

模型文件用 joblib（pickle）保存，只加载自己训练出来的文件。用 Teeth3DS（CC BY-NC-ND）训练的模型只能用于研究。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix, diags
from scipy.spatial import cKDTree

from ..fdi import QUADRANTS, Jaw
from ..orientation import normalize_orientation
from .base import Segmenter

N_CLASSES = 17
FEATURES = ["x", "y", "z", "nx", "ny", "nz", "curv1", "curv3", "h_local", "h_global", "arch_t", "arch_d", "arch_angle", "radius", "jaw"]


def fdi_to_class(labels: np.ndarray, jaw: Jaw) -> np.ndarray:
    right_q, left_q = QUADRANTS[jaw]
    q, p = labels // 10, labels % 10
    out = np.zeros(len(labels), dtype=np.int64)
    out[(q == right_q) & (p >= 1) & (p <= 8)] = p[(q == right_q) & (p >= 1) & (p <= 8)]
    out[(q == left_q) & (p >= 1) & (p <= 8)] = 8 + p[(q == left_q) & (p >= 1) & (p <= 8)]
    return out


def class_to_fdi(classes: np.ndarray, jaw: Jaw) -> np.ndarray:
    right_q, left_q = QUADRANTS[jaw]
    out = np.zeros(len(classes), dtype=np.int64)
    r = (classes >= 1) & (classes <= 8)
    l_ = classes >= 9
    out[r] = right_q * 10 + classes[r]
    out[l_] = left_q * 10 + classes[l_] - 8
    return out


def _adjacency(mesh: trimesh.Trimesh):
    e = mesh.edges_unique
    n = len(mesh.vertices)
    a = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    deg = np.asarray(a.sum(axis=1)).ravel()
    return diags(1.0 / np.maximum(deg, 1)) @ a  # 行归一化：邻居平均


def features(mesh: trimesh.Trimesh, jaw: Jaw) -> np.ndarray:
    """逐顶点特征（N, len(FEATURES)），在标准坐标系下计算。"""
    canon = normalize_orientation(mesh, jaw).apply(mesh)
    v = np.asarray(canon.vertices)
    n = np.asarray(canon.vertex_normals)
    avg = _adjacency(canon)
    curv1 = 1.0 - np.einsum("ij,ij->i", n, avg @ n)  # 相邻法向差：牙尖、牙颈的沟处大
    smooth = n
    for _ in range(3):
        smooth = avg @ smooth
    curv3 = 1.0 - np.einsum("ij,ij->i", n, smooth / np.maximum(np.linalg.norm(smooth, axis=1, keepdims=True), 1e-9))

    # 高度：相对整体（第 5 百分位）和相对局部最低点（2 mm 格子里的最低 z，再取邻域）
    z = v[:, 2]
    h_global = z - np.percentile(z, 5)
    cell = np.floor(v[:, :2] / 2.0).astype(np.int64)
    key = cell[:, 0] * 100003 + cell[:, 1]
    uniq, inv = np.unique(key, return_inverse=True)
    low = np.full(len(uniq), np.inf)
    np.minimum.at(low, inv, z)
    h_local = z - low[inv]

    # 牙弓：用高处的点（牙齿多在上面）拟合 y = a x² + b x + c
    top = v[z > np.percentile(z, 60)]
    coef = np.polyfit(top[:, 0], top[:, 1], 2)
    xs = np.linspace(v[:, 0].min() - 5, v[:, 0].max() + 5, 2001)
    curve = np.stack([xs, np.polyval(coef, xs)], axis=1)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    tree = cKDTree(curve)
    d, idx = tree.query(v[:, :2])
    mid = arc[np.argmin(np.abs(xs))]  # 中线（x = 0）处的弧长
    arch_t = arc[idx] - mid  # 沿牙弓的有符号弧长（mm），右侧为负
    # 到曲线的有符号距离：曲线法向朝牙弓外侧（凸的一侧）为正
    tangent = np.stack([np.ones_like(xs), 2 * coef[0] * xs + coef[1]], axis=1)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1) / np.linalg.norm(tangent, axis=1, keepdims=True)
    if normal[len(xs) // 2, 1] > 0:  # 中线处外侧在 -y（前方）
        normal = -normal
    arch_d = np.einsum("ij,ij->i", v[:, :2] - curve[idx], normal[idx])
    center_y = np.polyval(coef, 0.0) + 0.5 * (top[:, 1].max() - np.polyval(coef, 0.0))
    arch_angle = np.arctan2(v[:, 0], center_y - v[:, 1])
    radius = np.hypot(v[:, 0], v[:, 1] - center_y)
    jaw_f = np.full(len(v), 1.0 if jaw == "upper" else 0.0)
    return np.column_stack([v, n, curv1, curv3, h_local, h_global, arch_t, arch_d, arch_angle, radius, jaw_f]).astype(np.float32)


def _sample(labels: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """训练采样：牙齿和牙龈各一半（牙龈顶点占多数，不平衡会让模型偏向牙龈）。"""
    teeth, gum = np.nonzero(labels > 0)[0], np.nonzero(labels == 0)[0]
    k = n // 2
    pick = [rng.choice(teeth, min(k, len(teeth)), replace=False)] if len(teeth) else []
    pick.append(rng.choice(gum, min(n - k, len(gum)), replace=False))
    return np.concatenate(pick)


def _scan_rows(args):
    scan, n, seed = args
    try:
        if hasattr(scan, "load"):
            mesh, labels = scan.load()
        else:
            from ..io import load_labels, load_mesh

            mesh, labels = load_mesh(scan.mesh_path), load_labels(scan.gt_path)[0]
        x = features(mesh, scan.jaw)
        y = fdi_to_class(labels, scan.jaw)
        idx = _sample(y, n, np.random.default_rng(seed))
        return x[idx], y[idx]
    except Exception as exc:  # 个别扫描读不了不影响整体训练
        print(f"跳过 {scan.name}: {exc}", flush=True)
        return np.zeros((0, len(FEATURES)), np.float32), np.zeros(0, np.int64)


def train(scans, out: Path, per_scan: int = 6000, workers: int = 4, max_iter: int = 300, early_stopping: bool = True) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier

    t0 = time.time()
    jobs = [(s, per_scan, i) for i, s in enumerate(scans)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_scan_rows, jobs, chunksize=2))
    else:
        parts = [_scan_rows(job) for job in jobs]
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    t1 = time.time()
    print(f"特征：{len(scans)} 个扫描，{len(x)} 行，用时 {t1 - t0:.0f} 秒", flush=True)
    model = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=63, l2_regularization=1.0,
                                           early_stopping=early_stopping, validation_fraction=0.1, n_iter_no_change=15, random_state=0)
    model.fit(x, y)
    info = {"scans": len(scans), "rows": int(len(x)), "iterations": int(model.n_iter_), "features": FEATURES,
            "feature_seconds": round(t1 - t0), "fit_seconds": round(time.time() - t1)}
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "info": info}, out)
    print(json.dumps(info, ensure_ascii=False), flush=True)
    return info


class BaselineSegmenter(Segmenter):
    """梯度提升树基线。输出未经编号纠错的 FDI（流程里会再做纠错和去碎片）。"""

    name = "baseline_gbdt"
    postprocess = True

    def __init__(self, model_path: str | Path, smooth_iters: int = 6):
        import joblib

        self.model = joblib.load(model_path)["model"]
        self.smooth_iters = smooth_iters

    def predict(self, mesh: trimesh.Trimesh, jaw: Jaw) -> np.ndarray:
        proba = np.zeros((len(mesh.vertices), N_CLASSES))
        proba[:, self.model.classes_] = self.model.predict_proba(features(mesh, jaw))
        avg = _adjacency(mesh)
        for _ in range(self.smooth_iters):  # 沿网格平滑概率，去掉零散的错分点
            proba = 0.5 * proba + 0.5 * (avg @ proba)
        return class_to_fdi(np.argmax(proba, axis=1), jaw)


def main() -> None:
    from ..datasets import teeth3ds_zip_scans
    from ..evaluation import find_scans
    from ..io import load_mesh, save_labels

    p = argparse.ArgumentParser(description="CPU 分割基线（梯度提升树）")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--zips", type=Path, required=True, help="Teeth3DS 压缩包目录")
    t.add_argument("--split", type=Path, nargs="+", required=True, help="训练集划分文件（可多个）")
    t.add_argument("--extra-dir", type=Path, nargs="*", default=[], help="额外的训练数据目录（Teeth3DS 目录结构，如导出的 Poseidon3D 训练集），全部使用")
    t.add_argument("--scans", type=int, default=300, help="从 Teeth3DS 划分里随机抽多少个扫描（0 = 全部）")
    t.add_argument("--per-scan", type=int, default=6000)
    t.add_argument("--max-iter", type=int, default=300, help="梯度提升的最大轮数（有早停）")
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("predict")
    q.add_argument("--model", type=Path, required=True)
    q.add_argument("--data-dir", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--fix-numbering", action="store_true", help="输出前做编号纠错和去碎片（与流程一致）")
    args = p.parse_args()

    if args.cmd == "train":
        scans = [s for f in args.split for s in teeth3ds_zip_scans(args.zips, f)]
        rng = np.random.default_rng(0)
        scans = [scans[i] for i in rng.permutation(len(scans))[: args.scans or None]]
        for d in args.extra_dir:
            scans += find_scans(d, d)
        train(scans, args.out, args.per_scan, args.workers, max_iter=args.max_iter)
        return

    segmenter = BaselineSegmenter(args.model)
    scans = find_scans(args.data_dir, args.data_dir)
    for i, scan in enumerate(scans, 1):
        mesh = load_mesh(scan.mesh_path)
        t0 = time.time()
        labels = segmenter.predict(mesh, scan.jaw)
        if args.fix_numbering:
            from .numbering import correct_numbering
            from .postprocess import remove_small_islands

            labels = remove_small_islands(mesh, correct_numbering(mesh, labels, scan.jaw).labels)
        dest = args.out / scan.case / f"{scan.name}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        save_labels(dest, labels, scan.jaw, patient_id=scan.case)
        print(f"[{i}/{len(scans)}] {scan.name} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
