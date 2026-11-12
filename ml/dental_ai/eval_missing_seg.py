"""缺牙口扫上的分割评估：在带标注的真实口扫上模拟缺牙（synthetic.simulate_missing），看分割模型会不会把缺牙区认成牙。

    python -m dental_ai.eval_missing_seg --model ../models/segmentation/deep.pt --data-dir ../data/fair_test --out ../runs/missing_seg/deep

每个扫描两个场景：one（缺 1 颗）、multi（缺 2–3 颗，一半是相邻的）。分割后和流程里一样做编号纠错、去碎片，然后算：
- score / t_miou：挑战赛综合分和牙齿 mIoU（真值是去掉这些牙之后的标注）；
- phantom：被去掉的牙里，原位置有 20% 以上的顶点仍被标成牙的比例（“幻影牙”，越低越好）；
- detected：缺牙检测（detection.missing）报出了这颗牙的比例；
- exact：缺牙检测的结果和真值完全一致的扫描比例（不多报也不漏报）。
注意：模拟缺牙是把牙压平到牙龈高度，和真实的缺牙区（愈合后的牙槽嵴）不完全一样；真实缺牙病例看 Poseidon3D 的评估。
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

_SEG = None


def pick_missing(labels: np.ndarray, rng: np.random.Generator, n: int | None = None, keep: int = 6) -> tuple[int, ...]:
    """随机选要去掉的牙（不选第三磨牙）：n 颗，一半的情况下是同一象限里相邻的一段。"""
    present = sorted(int(f) for f in np.unique(labels) if f > 0 and f % 10 != 8)
    if len(present) <= keep:
        return ()
    n = int(n or rng.choice([1, 2, 3], p=[0.5, 0.3, 0.2]))
    n = min(n, len(present) - keep)
    if n > 1 and rng.random() < 0.5:
        start = int(rng.choice(present))
        run = [f for f in present if f // 10 == start // 10 and start % 10 <= f % 10 < start % 10 + n]
        if len(run) == n:
            return tuple(run)
    return tuple(int(f) for f in rng.choice(present, n, replace=False))


def _init(model: str, device: str | None):
    global _SEG
    from .segmentation.deep import DeepSegmenter

    _SEG = DeepSegmenter(model, device=device)


def _one(args) -> list[dict]:
    from .detection.missing import detect_missing
    from .evaluation import segmentation_metrics
    from .io import load_labels, load_mesh
    from .segmentation.numbering import correct_numbering
    from .segmentation.postprocess import remove_small_islands
    from .synthetic import simulate_missing

    mesh_path, gt_path, name, seed = args
    mesh0 = load_mesh(mesh_path)
    gt0, jaw = load_labels(gt_path)
    rng = np.random.default_rng(seed)
    rows = []
    for scenario, n in (("one", 1), ("multi", int(rng.integers(2, 4)))):
        teeth = pick_missing(gt0, rng, n)
        row = {"scan": name, "jaw": jaw, "scenario": scenario, "teeth": list(teeth)}
        try:
            mesh, gt = simulate_missing(mesh0, gt0, jaw, teeth)
            pred = _SEG.predict(mesh, jaw)
            pred = remove_small_islands(mesh, correct_numbering(mesh, pred, jaw).labels)
            metrics, _ = segmentation_metrics(gt, pred, mesh.vertices)
            frac = [float((pred[gt0 == f] != 0).mean()) for f in teeth]
            want = {m.fdi for m in detect_missing(mesh, gt, jaw)}
            got = {m.fdi for m in detect_missing(mesh, pred, jaw)}
            row.update(score=metrics["score"], t_miou=metrics["t_miou"], phantom=[f > 0.2 for f in frac], phantom_frac=frac,
                       detected=[f in got for f in teeth], exact=want == got, extra=sorted(got - want), missed=sorted(want - got))
        except Exception as exc:  # 跑完整批，单个失败记下来
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for scenario in ("one", "multi"):
        ok = [r for r in rows if r["scenario"] == scenario and "score" in r]
        if not ok:
            continue
        out[scenario] = {
            "scans": len(ok),
            "failed": sum(r["scenario"] == scenario and "error" in r for r in rows),
            "teeth_removed": sum(len(r["teeth"]) for r in ok),
            "score": round(float(np.mean([r["score"] for r in ok])), 4),
            "t_miou": round(float(np.mean([r["t_miou"] for r in ok])), 4),
            "phantom": round(float(np.mean([p for r in ok for p in r["phantom"]])), 4),
            "detected": round(float(np.mean([d for r in ok for d in r["detected"]])), 4),
            "exact": round(float(np.mean([r["exact"] for r in ok])), 4),
        }
    return out


def main() -> None:
    from .evaluation import find_scans

    p = argparse.ArgumentParser(description="缺牙口扫上的分割评估（模拟缺牙）")
    p.add_argument("--model", required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--device")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    scans = find_scans(args.data_dir, args.data_dir)[: args.limit]
    jobs = [(s.mesh_path, s.gt_path, s.name, i) for i, s in enumerate(scans)]
    rows = []
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.model, args.device)) as pool:
        for i, part in enumerate(pool.imap(_one, jobs), 1):
            rows += part
            if i % 20 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}]", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rows.json").write_text(json.dumps(rows))
    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
