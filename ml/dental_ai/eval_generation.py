"""缺失牙生成评估：在带标注的真实口扫上模拟缺牙（synthetic.simulate_missing），生成后与原来的真牙比较。

    python -m dental_ai.eval_generation --data-dir ../data/teeth3ds_test100 --out ../eval_generation [--templates ../models/templates]

每个扫描最多三个场景：
- single：缺一颗前磨牙或第一磨牙（两侧有邻牙，对侧同名牙在 → 用对侧镜像）；
- both_sides：左右同名的第一磨牙都缺（没有对侧可镜像 → 用模板）；
- free_end：最后一颗第二磨牙缺（游离端，按对侧镜像位置放置）。

指标（只看真牙牙冠上部 60%，牙颈附近两者的定义不同）：
- surf_mean / surf_p95：真牙表面点到生成牙表面的距离（mm）；
- position：牙冠上部水平质心的距离（mm）；
- top：生成牙最高点 − 真牙最高点（mm，咬合方向）；
- watertight：导出 STL 后按坐标焊接顶点，是否封闭。
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .evaluation import find_scans
from .fdi import QUADRANTS, mirror
from .geometry import estimate_arch_frame, tooth_mesh
from .io import load_labels, load_mesh
from .segmentation.base import Segmenter


class _Fixed(Segmenter):
    name = "label_file"
    postprocess = False

    def __init__(self, labels):
        self.labels = labels

    def predict(self, mesh, jaw):
        return self.labels


def scenarios(labels: np.ndarray, jaw: str, rng: np.random.Generator) -> list[tuple[str, tuple[int, ...]]]:
    present = {int(f) for f in np.unique(labels) if f > 0}
    out = []
    right_q, left_q = QUADRANTS[jaw]
    single = [q * 10 + p for q in (right_q, left_q) for p in (4, 5, 6)
              if q * 10 + p in present and q * 10 + p - 1 in present and q * 10 + p + 1 in present and mirror(q * 10 + p) in present]
    if single:
        out.append(("single", (int(rng.choice(single)),)))
    if {right_q * 10 + 6, left_q * 10 + 6, right_q * 10 + 5, right_q * 10 + 7, left_q * 10 + 5, left_q * 10 + 7} <= present:
        out.append(("both_sides", (right_q * 10 + 6, left_q * 10 + 6)))
    free = [q * 10 + 7 for q in (right_q, left_q) if q * 10 + 7 in present and q * 10 + 6 in present and q * 10 + 8 not in present]
    if free:
        out.append(("free_end", (int(rng.choice(free)),)))
    return out


def compare(solid: trimesh.Trimesh, real: trimesh.Trimesh, up: np.ndarray) -> dict:
    h = real.vertices @ up
    cut = np.percentile(h, 40)
    real_pts = trimesh.sample.sample_surface(real, 20000, seed=0)[0]
    real_pts = real_pts[real_pts @ up > cut]
    gen_pts = trimesh.sample.sample_surface(solid, 20000, seed=0)[0]
    d = cKDTree(gen_pts).query(real_pts)[0]
    gen_top = gen_pts[gen_pts @ up > cut]

    def flat(p):
        return p - np.outer(p @ up, up)

    position = np.linalg.norm(flat(gen_top.mean(axis=0)[None])[0] - flat(real_pts.mean(axis=0)[None])[0]) if len(gen_top) else np.nan
    return {
        "surf_mean": float(d.mean()),
        "surf_p95": float(np.percentile(d, 95)),
        "position": float(position),
        "top": float((solid.vertices @ up).max() - h.max()),
    }


def evaluate_scan(args) -> list[dict]:
    from .generation.mirror_template import MirrorTemplateGenerator
    from .pipeline import run
    from .synthetic import simulate_missing

    mesh_path, gt_path, name, templates, seed = args
    mesh0 = load_mesh(mesh_path)
    gt, jaw = load_labels(gt_path)
    rng = np.random.default_rng(seed)
    rows = []
    try:
        up = estimate_arch_frame(mesh0, gt, jaw).occlusal
    except ValueError:
        return rows
    for kind, teeth in scenarios(gt, jaw, rng):
        row = {"scan": name, "jaw": jaw, "scenario": kind, "teeth": " ".join(map(str, teeth))}
        t0 = time.time()
        try:
            mesh, labels = simulate_missing(mesh0, gt, jaw, teeth)
            with tempfile.TemporaryDirectory() as out:
                generator = MirrorTemplateGenerator(template_dir=templates) if templates else None
                report = run(mesh, jaw, _Fixed(labels), out, generator=generator)
                done = {g["fdi"]: g for g in report["generated"]}
                for fdi in teeth:
                    r = dict(row, fdi=fdi)
                    if fdi not in done:
                        r["error"] = "未生成：" + "; ".join(w for w in report["warnings"] if str(fdi) in w)[:200]
                        rows.append(r)
                        continue
                    g = done[fdi]
                    solid = trimesh.load(Path(out) / g["file"])  # 默认按坐标焊接顶点
                    r.update(compare(solid, tooth_mesh(mesh0, gt, fdi), up))
                    r.update(source=g.get("source"), placement=g.get("placement"), watertight=bool(solid.is_watertight),
                             seconds=round(time.time() - t0, 1))
                    rows.append(r)
        except Exception as exc:  # 评估要跑完整批数据，单个失败记录下来
            rows.append(dict(row, error=f"{type(exc).__name__}: {exc}"[:200]))
    return rows


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for kind in ("single", "both_sides", "free_end"):
        sub = [r for r in rows if r["scenario"] == kind]
        ok = [r for r in sub if "surf_mean" in r]
        if not sub:
            continue

        def med(k):
            return round(float(np.nanmedian([r[k] for r in ok])), 3) if ok else None

        summary[kind] = {
            "teeth": len(sub),
            "generated": len(ok),
            "watertight": sum(bool(r.get("watertight")) for r in ok),
            "surf_mean_median": med("surf_mean"),
            "surf_p95_median": med("surf_p95"),
            "position_median": med("position"),
            "top_abs_median": round(float(np.median([abs(r["top"]) for r in ok])), 3) if ok else None,
            "sources": sorted({str(r.get("source", "")).split(":")[0] for r in ok}),
        }
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="缺失牙生成评估（在真实口扫上模拟缺牙，与真牙比较）")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--gt-dir", type=Path, help="默认与 --data-dir 相同")
    p.add_argument("--templates", type=Path, help="标准牙模板库目录（不填用内置模板）")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("eval_generation"))
    args = p.parse_args()

    scans = find_scans(args.data_dir, args.gt_dir or args.data_dir)[: args.limit]
    print(f"找到 {len(scans)} 个扫描", flush=True)
    jobs = [(s.mesh_path, s.gt_path, s.name, args.templates, i) for i, s in enumerate(scans)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, part in enumerate(pool.map(evaluate_scan, jobs), 1):
            rows += part
            if i % 10 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}]", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("scan", "jaw", "scenario", "teeth", "fdi"), k))
    with open(args.out / "generation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    (args.out / "generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
