"""牙冠质量评估：在咬合对齐的上下颌口扫上，用同一套评分卡（crown_quality.py）量天然牙和生成的牙冠。

    python -m dental_ai.eval_crown --zip ~/zyt/dental_data/Bite2Text.zip --truth ../runs/real_missing/truth.json \\
        --seg-model ../models/segmentation/deep.pt --gen-model ../models/generation/toothgen_open.pt --out ../runs/crown_quality

数据：Bite2Text（每位患者一对咬合对齐的上下颌 STL，没有逐牙标注）。牙的范围和牙号用本项目的分割模型自动标，
所以“天然牙”一行里也带着分割的误差（边界差零点几毫米）；给了 --truth 时只用报告里没提缺牙的患者。

每个颌随机选几颗两侧都有邻牙的前磨牙 / 第一磨牙，每颗牙出三行：
- natural：原来的真牙，量出来的分布就是及格线；
- generated：把这颗牙去掉（synthetic.simulate_missing），走完整流程生成，对颌传给生成器；
- generated_no_antagonist：同上，但不给对颌（现在后端没有对颌时的情况）。

汇总里每个指标给出中位数和 5%–95% 分位；生成的牙冠另外给出“落在天然牙 5%–95% 范围内”的比例。
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import trimesh

METRICS = [
    "mesial_gap", "distal_gap", "mesial_contact_area", "distal_contact_area",
    "occl_clearance", "occl_penetration_area", "occl_contact_area", "occl_contacts",
    "relief", "detail", "height_vs_neighbors", "arch_offset",
    "ref_chamfer_l1", "ref_surf_p95", "ref_fscore_0.3", "ref_position", "ref_top",
]
_STATE: dict = {}


def _load(zip_path: str, pid: str, jaw: str) -> trimesh.Trimesh:
    mesh = trimesh.load(io.BytesIO(zipfile.ZipFile(zip_path).read(f"{pid}/ios/ios_{jaw}.stl")), file_type="stl")
    mesh.remove_unreferenced_vertices()  # 个别口扫有没被三角形引用的顶点，模拟缺牙时顶点数会对不上
    return mesh


def _labels(mesh, pid: str, jaw: str, seg_model: str, cache: Path) -> np.ndarray:
    from .pipeline import correct_numbering, merge_split_teeth, remove_small_islands

    f = cache / f"{pid}_{jaw}.npy"
    if f.exists():
        return np.load(f).astype(int)
    if "seg" not in _STATE:
        from .segmentation.deep import DeepSegmenter

        _STATE["seg"] = DeepSegmenter(seg_model)
    pts = np.asarray(mesh.vertices)
    labels = remove_small_islands(mesh, correct_numbering(mesh, _STATE["seg"].predict(mesh, jaw), jaw, canonical_vertices=pts).labels)
    labels = merge_split_teeth(mesh, labels, jaw, points=pts)[0]
    np.save(f, labels.astype(np.int16))
    return labels


def _one(args) -> list[dict]:
    from .crown_quality import _crop, score_crown
    from .eval_generation import _Fixed
    from .generation.learned import LearnedGenerator
    from .generation.mirror_template import MirrorTemplateGenerator
    from .geometry import estimate_arch_frame, tooth_mesh
    from .pipeline import run
    from .synthetic import simulate_missing

    zip_path, pid, seg_model, gen_model, out, per_jaw, seed = args
    rows: list[dict] = []
    try:
        meshes = {jaw: _load(zip_path, pid, jaw) for jaw in ("upper", "lower")}
        labels = {jaw: _labels(meshes[jaw], pid, jaw, seg_model, Path(out) / "labels") for jaw in meshes}
    except Exception as exc:
        return [{"patient": pid, "error": f"{type(exc).__name__}: {exc}"[:200]}]
    rng = np.random.default_rng(seed)
    for jaw, other in (("upper", "lower"), ("lower", "upper")):
        mesh, lab = meshes[jaw], labels[jaw]
        present = {int(f) for f in np.unique(lab) if f}
        cand = sorted(f for f in present if f % 10 in (4, 5, 6) and f - 1 in present and f + 1 in present)
        try:
            up = estimate_arch_frame(mesh, lab, jaw).occlusal
        except ValueError:
            continue
        for fdi in rng.permutation(cand)[:per_jaw].tolist():
            base = {"patient": pid, "jaw": jaw, "fdi": fdi}
            try:
                real = tooth_mesh(mesh, lab, fdi)
                mesial, distal = tooth_mesh(mesh, lab, fdi - 1), tooth_mesh(mesh, lab, fdi + 1)
                ant = _crop(meshes[other], np.asarray(real.vertices), 12.0)
                rows.append({**base, "kind": "natural", **score_crown(real, up, mesial, distal, ant)})
                mesh2, lab2 = simulate_missing(mesh, lab, jaw, (fdi,))
                for kind, antagonist in (("generated", ant), ("generated_no_antagonist", None)):
                    with tempfile.TemporaryDirectory() as tmp:
                        gen = LearnedGenerator(gen_model, fallback=MirrorTemplateGenerator())
                        report = run(mesh2, jaw, _Fixed(lab2), tmp, generator=gen, antagonist=antagonist)
                        done = {g["fdi"]: g for g in report["generated"]}
                        if fdi not in done:
                            rows.append({**base, "kind": kind, "error": "未生成：" + "; ".join(w for w in report["warnings"] if str(fdi) in w)[:160]})
                            continue
                        solid = trimesh.load(Path(tmp) / done[fdi]["file"])
                    rows.append({**base, "kind": kind, "source": done[fdi].get("source"), **score_crown(solid, up, mesial, distal, ant, reference=real)})
            except Exception as exc:  # 评估要跑完整批，单颗失败记下来
                rows.append({**base, "kind": "error", "error": f"{type(exc).__name__}: {exc}"[:200]})
    return rows


def summarize(rows: list[dict]) -> dict:
    def values(kind, key):
        v = np.array([r[key] for r in rows if r.get("kind") == kind and r.get(key) is not None], float)
        return v[np.isfinite(v)]

    summary: dict = {"teeth": {k: sum(r.get("kind") == k and "error" not in r for r in rows) for k in ("natural", "generated", "generated_no_antagonist")},
                     "errors": sum("error" in r for r in rows), "metrics": {}}
    for key in METRICS:
        nat = values("natural", key)
        block = {}
        lo, hi = (np.percentile(nat, [5, 95]) if len(nat) else (None, None))
        for kind in ("natural", "generated", "generated_no_antagonist"):
            v = values(kind, key)
            if not len(v):
                continue
            v = v[np.isfinite(v)]
            block[kind] = {"median": round(float(np.median(v)), 3), "p5": round(float(np.percentile(v, 5)), 3), "p95": round(float(np.percentile(v, 95)), 3)}
            if kind != "natural" and lo is not None:
                block[kind]["within_natural_range"] = round(float(np.mean((v >= lo) & (v <= hi))), 3)
        if block:
            summary["metrics"][key] = block
    # 几个一眼能看懂的比例
    def rate(kind, fn):
        sub = [r for r in rows if r.get("kind") == kind and "error" not in r]
        return round(float(np.mean([fn(r) for r in sub])), 3) if sub else None

    checks = {
        "穿入对颌超过 0.1 mm": lambda r: r.get("occl_clearance") is not None and r["occl_clearance"] < -0.1,
        "和对颌没有接触（间隙 > 0.2 mm）": lambda r: r.get("occl_clearance") is None or r["occl_clearance"] > 0.2,
        "至少一侧邻接有缝（> 0.15 mm）": lambda r: any(r.get(k) is not None and r[k] > 0.15 for k in ("mesial_gap", "distal_gap")),
        "至少一侧挤进邻牙（< -0.15 mm）": lambda r: any(r.get(k) is not None and r[k] < -0.15 for k in ("mesial_gap", "distal_gap")),
        "不封闭": lambda r: not r.get("watertight"),
    }
    summary["checks"] = {name: {k: rate(k, fn) for k in ("natural", "generated", "generated_no_antagonist")} for name, fn in checks.items()}
    return summary


def main() -> None:
    import multiprocessing as mp

    p = argparse.ArgumentParser(description="牙冠质量评分卡：天然牙 vs 生成的牙冠（Bite2Text）")
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--truth", type=Path, help="eval_real_missing extract 的输出；给了就只用报告里没提缺牙的患者")
    p.add_argument("--seg-model", required=True)
    p.add_argument("--gen-model", required=True)
    p.add_argument("--patients", type=int, default=60)
    p.add_argument("--per-jaw", type=int, default=2)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    names = set(zipfile.ZipFile(args.zip).namelist())
    pids = sorted({n.split("/")[0] for n in names if n.endswith("ios_upper.stl") and n.replace("upper", "lower") in names})
    if args.truth:
        truth = json.loads(args.truth.read_text())
        pids = [q for q in pids if q in truth and not truth[q]["missing"] and not truth[q]["other_mentions"] and not truth[q]["sentences"]]
    rng = np.random.default_rng(0)
    pids = [pids[i] for i in rng.permutation(len(pids))[: args.patients]]
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    jobs = [(str(args.zip), q, args.seg_model, args.gen_model, str(args.out), args.per_jaw, i) for i, q in enumerate(pids)]
    print(f"{len(pids)} 位患者", flush=True)
    rows: list[dict] = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, part in enumerate(pool.imap_unordered(_one, jobs), 1):
            rows += part
            if i % 5 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}]", flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("patient", "jaw", "fdi", "kind"), k))
    with open(args.out / "crowns.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
