"""真实代型（teethPreparationData 52 个，.vtp 带颈缘线标注）上比几何颈缘线的参数：分割给的起点 = 真值线平移 offset mm（模拟分割误差）。
看“先验”（往起点拉）和“搜索半径”在真实预备体上是不是还有用。"""
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
R = Path.home() / "zyt/dental_3d_ai"
sys.path.insert(0, str(R / "ml"))
CONFIGS = {"默认": {}, "先验 0.5": {"prior_weight": 0.5}, "先验 2": {"prior_weight": 2.0}, "半径 1.0": {"search_radius": 1.0},
           "半径 1.0 + 先验 0.5": {"search_radius": 1.0, "prior_weight": 0.5}, "平滑 4": {"smooth_weight": 4.0}}


def one(path):
    from dental_ai.eval_margin import evaluate_case

    return {name: evaluate_case(Path(path), kw) for name, kw in CONFIGS.items()}


if __name__ == "__main__":
    import multiprocessing as mp

    files = sorted((R / "data/prep_margin/training Data").glob("*.vtp")) + sorted((R / "data/prep_margin/test Data").glob("*.vtp"))
    with mp.get_context("spawn").Pool(40) as pool:
        res = pool.map(one, [str(f) for f in files])
    print(f"{len(files)} 个真实代型")
    offs = sorted({r["offset"] for x in res for r in x["默认"]})
    for name in CONFIGS:
        line = []
        for o in offs:
            v = [r.get("det_mean") for x in res for r in x[name] if r["offset"] == o and "error" not in r]
            init = [r.get("init_mean") for x in res for r in x[name] if r["offset"] == o and "error" not in r]
            line.append(f"起点偏 {o:+.1f}: {np.median(v):.3f}（起点 {np.median(init):.2f}）")
        print(f"{name:14s} " + " | ".join(line))
    (R / "runs/margin_die_sweep.json").write_text(json.dumps(res, default=float))
