"""几何颈缘线调参：在预备体评分卡那 115 颗模拟预备体上，只算颈缘线（不生成牙冠），每组参数的误差（平均 / 95% / 超过 1 mm 的比例）。
一次改一个参数，和默认值比。CPU 多进程，每颗牙一个任务、所有参数组都在里面算。"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
R = Path.home() / "zyt/dental_3d_ai"
sys.path.insert(0, str(R / "ml"))

CONFIGS = {
    "默认": {},
    "搜索半径 1.0": {"search_radius": 1.0},
    "搜索半径 2.0": {"search_radius": 2.0},
    "平滑 1": {"smooth_weight": 1.0},
    "平滑 4": {"smooth_weight": 4.0},
    "滑动 0.5": {"slide_weight": 0.5},
    "滑动 2": {"slide_weight": 2.0},
    "先验 0.5": {"prior_weight": 0.5},
    "先验 2": {"prior_weight": 2.0},
    "高度差 0.8": {"max_height_diff": 0.8},
    "高度差 1.8": {"max_height_diff": 1.8},
    "站点 200": {"stations": 200},
    "候选 80": {"candidates": 80},
    "起点候选 16": {"start_candidates": 16},
}


def one(args):
    pid, jaw, fdi = args
    from scipy.spatial import cKDTree

    from dental_ai.eval_crown import _load
    from dental_ai.geometry import estimate_arch_frame
    from dental_ai.margin.detect import detect_margin, edge_sharpness
    from dental_ai.synthetic import simulate_preparation

    try:
        mesh = _load(str(Path.home() / "zyt/dental_data/Bite2Text.zip"), pid, jaw)
        lab = np.load(R / f"runs/prep_card_base2/labels/{pid}_{jaw}.npy").astype(int)
        prep, truth = simulate_preparation(mesh, lab, jaw, fdi)
        up = estimate_arch_frame(prep, lab, jaw).occlusal
        sharp = edge_sharpness(prep)
        out = {}
        for name, kw in CONFIGS.items():
            try:
                pts = np.asarray(detect_margin(prep, lab, fdi, up=up, sharpness=sharp, **kw).points)
                d1, d2 = cKDTree(truth).query(pts)[0], cKDTree(pts).query(truth)[0]
                out[name] = ((d1.mean() + d2.mean()) / 2, float(np.percentile(np.concatenate([d1, d2]), 95)))
            except Exception:  # noqa: BLE001
                out[name] = (5.0, 5.0)
        return out
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:100]}


if __name__ == "__main__":
    import multiprocessing as mp

    rows = [r for r in csv.DictReader(open(R / "runs/prep_card_base2/crowns.csv")) if r.get("kind") == "auto" and not r.get("error")]
    jobs = [(r["patient"], r["jaw"], int(r["fdi"])) for r in rows]
    with mp.get_context("spawn").Pool(40) as pool:
        res = [r for r in pool.map(one, jobs) if "error" not in r]
    print(f"{len(res)} 颗")
    for name in CONFIGS:
        m = np.array([r[name][0] for r in res])
        p = np.array([r[name][1] for r in res])
        print(f"{name:10s} 平均 {np.mean(m):.3f}  中位 {np.median(m):.3f}  95% {np.median(p):.3f}  >1mm {np.mean(m > 1):.3f}  失败 {np.mean(m >= 5):.3f}")
    (R / "runs/margin_sweep.json").write_text(json.dumps(res))
