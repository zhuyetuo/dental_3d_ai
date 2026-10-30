"""分割评估：Teeth3DS / 3DTeethSeg'22 格式的数据集，以及自有数据集。

三组指标：

1. 挑战赛官方指标（严格复刻 3DTeethSeg22_challenge/evaluation/evaluation.py，commit 6220bfa）：
    tla    exp(-TLA)。TLA = 真值牙质心与匹配的预测牙质心的距离，按“牙齿尺寸”逐轴归一化后取模，再对牙取平均；
           真值牙和预测牙按质心距离用匈牙利算法一一匹配；没有匹配上的真值牙记 5 倍惩罚
    tsa    牙齿 / 牙龈二分类的 micro-F1（等于顶点二分类准确率）
    tir    匹配上的牙中，归一化距离 < 0.5 且编号相同的比例
    score  (tla + tsa + tir) / 3
   其中“牙齿尺寸”沿用官方实现：每个坐标轴上 sqrt(Σ(顶点 - 质心)²)。
   预测实例内部编号不一致时，官方会把该实例整体当作牙龈；本模块在预测没有实例信息时用编号当作实例。

2. TGNet 口径（复刻 ToothGroupNetwork/eval_visualize_results.py，可以和 TGNet README 的数字对比）：
    inst_iou / inst_f1 / inst_tir  每个预测实例按多数票匹配真值牙后的 IoU、F1、编号正确率

3. 论文常用口径：
    t_miou       每颗真值牙与同编号预测区域的 IoU，取平均（3DTeethSAM、TSegAgent 使用）
    all_correct  整副牙列全部识别正确：每颗真值牙都匹配上、距离 < 0.5、编号相同（TSegAgent 的“整副牙列正确率”）

命令行：
    python -m dental_ai.evaluation --data-dir <obj 目录> --gt-dir <标注目录> --pred-dir <预测目录> --out results/
    python -m dental_ai.evaluation --data-dir ... --gt-dir ... --segmenter-url http://localhost:8001 --orient --out results/
    python -m dental_ai.evaluation --data-dir ... --gt-dir ... --check-orientation --out results/
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .io import load_labels, load_mesh

OFFICIAL = ["tla", "tsa", "tir", "score"]
METRICS = OFFICIAL + ["inst_iou", "inst_f1", "inst_tir", "t_miou", "all_correct"]


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _instances(labels: np.ndarray, instances: np.ndarray | None, vertices: np.ndarray, with_size: bool) -> dict:
    """{实例号: {label, centroid, size}}。编号不唯一的实例按官方规则丢弃。"""
    inst = labels if instances is None else instances
    out = {}
    for i in np.unique(inst):
        if i == 0:
            continue
        mask = inst == i
        lab = np.unique(labels[mask])
        if len(lab) != 1 or lab[0] == 0:
            continue
        pts = vertices[mask]
        c = pts.mean(axis=0)
        info = {"label": int(lab[0]), "centroid": c}
        if with_size:
            info["size"] = np.sqrt(((pts - c) ** 2).sum(axis=0))
        out[int(i)] = info
    return out


def challenge_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    vertices: np.ndarray,
    gt_instances: np.ndarray | None = None,
    pred_instances: np.ndarray | None = None,
) -> dict[str, float]:
    g = _instances(gt, gt_instances, vertices, with_size=True)
    p = _instances(pred, pred_instances, vertices, with_size=False)
    gk, pk = list(g), list(p)
    match: dict[int, int] = {}
    if gk and pk:
        cost = cdist([g[k]["centroid"] for k in gk], [p[k]["centroid"] for k in pk])
        rows, cols = linear_sum_assignment(cost)
        match = {gk[r]: pk[c] for r, c in zip(rows, cols)}

    def norm_dist(gi: int) -> float:
        return float(np.linalg.norm((g[gi]["centroid"] - p[match[gi]]["centroid"]) / g[gi]["size"]))

    tla_raw = 0.0
    for gi in gk:
        tla_raw += norm_dist(gi) if gi in match else 5 * float(np.linalg.norm(g[gi]["size"]))
    tla_raw = _safe_div(tla_raw, len(gk))

    # 官方的 TSA：把非零实例当作牙齿的二分类 micro-F1；被丢弃的预测实例按牙龈处理
    p_inst = pred if pred_instances is None else pred_instances
    pred_fg = np.isin(p_inst, list(p)) if p else np.zeros(len(pred), bool)
    g_inst = gt if gt_instances is None else gt_instances
    tsa = float(np.mean((g_inst != 0) == pred_fg))

    correct = [gi for gi in match if norm_dist(gi) < 0.5 and g[gi]["label"] == p[match[gi]]["label"]]
    tir = _safe_div(len(correct), len(match))
    tla = float(np.exp(-tla_raw)) if gk else 0.0
    return {
        "tla": tla,
        "tsa": tsa,
        "tir": tir,
        "score": (tla + tsa + tir) / 3,
        "all_correct": float(bool(gk) and len(correct) == len(gk)),
    }


def instance_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """按预测实例计算，复刻 TGNet 的 cal_metric（预测编号同时作为实例）。"""
    names = [int(v) for v in np.unique(pred) if v != 0]
    if not names:
        return {"inst_iou": 0.0, "inst_f1": 0.0, "inst_tir": 0.0}
    iou = f1 = tir = 0.0
    for name in names:
        ins = pred == name
        values, counts = np.unique(gt[ins], return_counts=True)
        gt_name = values[np.argmax(counts)]
        g = gt == gt_name
        tp = np.count_nonzero(g & ins)
        fp = np.count_nonzero(~g & ins)
        fn = np.count_nonzero(g & ~ins)
        precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
        f1 += _safe_div(2 * precision * recall, precision + recall)
        iou += _safe_div(tp, tp + fp + fn)
        tir += float(name == gt_name)
    n = len(names)
    return {"inst_iou": iou / n, "inst_f1": f1 / n, "inst_tir": tir / n}


def per_tooth_iou(gt: np.ndarray, pred: np.ndarray) -> dict[int, float]:
    out = {}
    for f in np.unique(gt):
        if f == 0:
            continue
        g, p = gt == f, pred == f
        out[int(f)] = _safe_div(np.count_nonzero(g & p), np.count_nonzero(g | p))
    return out


def segmentation_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    vertices: np.ndarray,
    gt_instances: np.ndarray | None = None,
    pred_instances: np.ndarray | None = None,
) -> tuple[dict[str, float], dict[int, float]]:
    gt, pred, vertices = np.asarray(gt), np.asarray(pred), np.asarray(vertices)
    if gt.shape != pred.shape:
        raise ValueError(f"标签长度不一致：真值 {gt.shape}，预测 {pred.shape}")
    ious = per_tooth_iou(gt, pred)
    metrics = {
        **challenge_metrics(gt, pred, vertices, gt_instances, pred_instances),
        **instance_metrics(gt, pred),
        "t_miou": float(np.mean(list(ious.values()))) if ious else 0.0,
    }
    return metrics, ious


# ---------------------------------------------------------------- 数据集

@dataclass
class Scan:
    case: str
    jaw: str
    mesh_path: Path
    gt_path: Path

    @property
    def name(self) -> str:
        return f"{self.case}_{self.jaw}"


def find_scans(data_dir: Path, gt_dir: Path, split: Path | None = None) -> list[Scan]:
    """Teeth3DS 目录结构：<data>/<病例>/<病例>_<upper|lower>.obj 与 <gt>/<病例>/<病例>_<upper|lower>.json。"""
    # 划分文件每行可以是病例名（挑战赛划分），也可以是“病例_颌位”（Teeth3DS 官方划分）
    allowed = None
    if split:
        allowed = {line.strip() for line in split.read_text().splitlines() if line.strip()}
    scans = []
    for obj in sorted(data_dir.rglob("*.obj")):
        stem = obj.stem
        if "_" not in stem:
            continue
        case, jaw = stem.rsplit("_", 1)
        if jaw not in ("upper", "lower") or (allowed is not None and case not in allowed and stem not in allowed):
            continue
        candidates = [gt_dir / case / f"{stem}.json", gt_dir / f"{stem}.json"]
        gt = next((c for c in candidates if c.exists()), None)
        if gt:
            scans.append(Scan(case, jaw, obj, gt))
    return scans


def load_instances(path: Path) -> np.ndarray | None:
    data = json.loads(Path(path).read_text())
    return np.asarray(data["instances"]) if "instances" in data else None


def _summarize(rows: list[dict]) -> dict:
    def mean(sub):
        return {m: float(np.mean([r[m] for r in sub])) for m in METRICS} if sub else {}

    return {
        "n_scans": len(rows),
        "all": mean(rows),
        "upper": mean([r for r in rows if r["jaw"] == "upper"]),
        "lower": mean([r for r in rows if r["jaw"] == "lower"]),
    }


def evaluate(scans: list[Scan], predict, out_dir: Path) -> dict:
    """predict(scan, mesh) -> 逐顶点预测标签。逐个病例评估并写出 per_scan.csv、per_tooth_iou.csv、summary.json。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, tooth_rows, failures = [], [], []
    for i, scan in enumerate(scans, 1):
        try:
            mesh = load_mesh(scan.mesh_path)
            gt, _ = load_labels(scan.gt_path)
            gt_instances = load_instances(scan.gt_path)
            t = time.time()
            pred = predict(scan, mesh)
            seconds = time.time() - t
            metrics, per_tooth = segmentation_metrics(gt, pred, mesh.vertices, gt_instances)
        except Exception as exc:  # 单个病例失败不影响整体评估，单独记录
            failures.append({"scan": scan.name, "error": str(exc)})
            print(f"[{i}/{len(scans)}] {scan.name} 失败: {exc}")
            continue
        rows.append({"scan": scan.name, "jaw": scan.jaw, "seconds": round(seconds, 2), **metrics})
        tooth_rows += [{"scan": scan.name, "fdi": f, "iou": v} for f, v in per_tooth.items()]
        print(f"[{i}/{len(scans)}] {scan.name} t_miou={metrics['t_miou']:.4f} tir={metrics['tir']:.4f}")

    if rows:
        with open(out_dir / "per_scan.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if tooth_rows:
        with open(out_dir / "per_tooth_iou.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["scan", "fdi", "iou"])
            w.writeheader()
            w.writerows(tooth_rows)

    summary = _summarize(rows)
    # 官方规则：失败的扫描三项指标都记 0
    n_total = len(rows) + len(failures)
    summary["official_with_failures"] = (
        {m: float(sum(r[m] for r in rows) / n_total) for m in OFFICIAL} if n_total else {}
    )
    by_fdi: dict[int, list[float]] = {}
    for r in tooth_rows:
        by_fdi.setdefault(r["fdi"], []).append(r["iou"])
    summary["per_fdi_iou"] = {str(k): float(np.mean(v)) for k, v in sorted(by_fdi.items())}
    summary["failures"] = failures
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# ---------------------------------------------------------------- 朝向检查

def rotation_angle(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1.0, 1.0))))


def check_orientation(scans: list[Scan], out_dir: Path, trials: int = 3, seed: int = 0) -> dict:
    """在已标准化的数据集（如 Teeth3DS）上验证朝向归一化：
    1. 用真值标签检查数据本身是否符合约定的坐标系（左侧象限在 +X、磨牙在 +Y、牙冠方向）；
    2. 随机旋转后做归一化，统计恢复误差（角度）。"""
    from scipy.spatial.transform import Rotation

    from .fdi import QUADRANTS
    from .orientation import normalize_orientation

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for i, scan in enumerate(scans, 1):
        mesh = load_mesh(scan.mesh_path)
        gt, _ = load_labels(scan.gt_path)
        v = mesh.vertices
        right_q, left_q = QUADRANTS[scan.jaw]
        left = v[(gt // 10 == left_q)]
        right = v[(gt // 10 == right_q)]
        incisors = v[np.isin(gt, [right_q * 10 + 1, left_q * 10 + 1])]
        molars = v[np.isin(gt, [right_q * 10 + 6, left_q * 10 + 6, right_q * 10 + 7, left_q * 10 + 7])]
        teeth_z, gum_z = v[gt != 0][:, 2].mean(), v[gt == 0][:, 2].mean()
        crown_up = teeth_z > gum_z
        row = {
            "scan": scan.name,
            "left_is_pos_x": bool(len(left) and len(right) and left[:, 0].mean() > right[:, 0].mean()),
            "molars_pos_y": bool(len(incisors) and len(molars) and molars[:, 1].mean() > incisors[:, 1].mean()),
            "crown_dir_ok": bool(crown_up == (scan.jaw == "lower")),
        }
        errors = []
        for _ in range(trials):
            rot = Rotation.random(random_state=int(rng.integers(1 << 31))).as_matrix()
            moved = mesh.copy()
            transform = np.eye(4)
            transform[:3, :3] = rot
            moved.apply_transform(transform)
            result = normalize_orientation(moved, scan.jaw)
            errors.append(rotation_angle(result.transform[:3, :3] @ rot))
        row["angle_err_mean"] = float(np.mean(errors))
        row["angle_err_max"] = float(np.max(errors))
        rows.append(row)
        print(f"[{i}/{len(scans)}] {scan.name} 约定检查 {row['left_is_pos_x']}/{row['molars_pos_y']}/{row['crown_dir_ok']} 恢复误差 {row['angle_err_max']:.1f}°")

    if rows:
        with open(out_dir / "orientation.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    errs = np.array([r["angle_err_max"] for r in rows]) if rows else np.zeros(0)
    summary = {
        "n_scans": len(rows),
        "convention_ok_rate": float(np.mean([r["left_is_pos_x"] and r["molars_pos_y"] and r["crown_dir_ok"] for r in rows])) if rows else 0.0,
        "angle_err_median": float(np.median(errs)) if len(errs) else 0.0,
        "within_10deg_rate": float(np.mean(errs <= 10)) if len(errs) else 0.0,
        "flipped_rate": float(np.mean(errs >= 90)) if len(errs) else 0.0,
    }
    (out_dir / "orientation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# ---------------------------------------------------------------- 命令行

def _print_summary(summary: dict) -> None:
    print(f"\n扫描数: {summary['n_scans']}   失败: {len(summary.get('failures', []))}")
    print(f"{'指标':<12}" + "".join(f"{k:>10}" for k in ("all", "upper", "lower")))
    for m in METRICS:
        vals = [summary[k].get(m) if summary[k] else None for k in ("all", "upper", "lower")]
        print(f"{m:<12}" + "".join(f"{v * 100:>9.2f}%" if v is not None else f"{'-':>10}" for v in vals))


def main() -> None:
    p = argparse.ArgumentParser(description="分割评估（Teeth3DS 格式）")
    p.add_argument("--data-dir", required=True, type=Path, help="OBJ 目录")
    p.add_argument("--gt-dir", required=True, type=Path, help="真值标注目录")
    p.add_argument("--split", type=Path, help="病例名列表（每行一个），如 Teeth3DS 的 base_name_test_fold.txt")
    p.add_argument("--limit", type=int, help="只评估前 N 个扫描（快速检查）")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--pred-dir", type=Path, help="已有预测结果目录（TGNet 输出格式）")
    src.add_argument("--segmenter-url", help="分割服务地址，逐个病例调用")
    src.add_argument("--check-orientation", action="store_true", help="只验证朝向归一化，不评估分割")
    p.add_argument("--orient", action="store_true", help="调用分割服务前先做朝向归一化")
    p.add_argument("--fix-numbering", action="store_true", help="对预测结果做编号纠错后再评估（用来衡量纠错带来的提升）")
    p.add_argument("--out", type=Path, default=Path("eval_results"))
    args = p.parse_args()

    scans = find_scans(args.data_dir, args.gt_dir, args.split)
    if args.limit:
        scans = scans[: args.limit]
    if not scans:
        p.error("没有找到 OBJ 与标注能对上的扫描，请检查目录结构")
    print(f"找到 {len(scans)} 个扫描")

    if args.check_orientation:
        print(json.dumps(check_orientation(scans, args.out), ensure_ascii=False, indent=2))
        return

    if args.pred_dir:
        def predict(scan: Scan, mesh):
            for c in (args.pred_dir / f"{scan.name}.json", args.pred_dir / scan.case / f"{scan.name}.json"):
                if c.exists():
                    return load_labels(c)[0]
            raise FileNotFoundError(f"没有预测结果 {scan.name}.json")
    elif args.segmenter_url:
        from .orientation import OrientedSegmenter
        from .segmentation.remote import RemoteSegmenter

        segmenter = RemoteSegmenter(args.segmenter_url)
        if args.orient:
            segmenter = OrientedSegmenter(segmenter)

        def predict(scan: Scan, mesh):
            return segmenter.predict(mesh, scan.jaw)
    else:
        p.error("需要 --pred-dir、--segmenter-url 或 --check-orientation 之一")

    if args.fix_numbering:
        from .segmentation.numbering import correct_numbering

        base_predict = predict

        def predict(scan: Scan, mesh):  # noqa: F811
            return correct_numbering(mesh, base_predict(scan, mesh), scan.jaw).labels

    summary = evaluate(scans, predict, args.out)
    _print_summary(summary)
    print(f"\n结果已写入 {args.out}/")


if __name__ == "__main__":
    main()
