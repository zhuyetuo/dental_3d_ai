"""颈缘线融合的快速对比（只算颈缘线，不生成牙冠）：和预备体评分卡（--prep-sim real）同样的病人、同样的牙、同样的模拟预备体。
几种做法：几何 / 学习式 / 学习式当初始线再贴棱（snap）/ 两条线差太多就用几何（gate）/ 按棱强度选。"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _err(line, truth):
    return float((cKDTree(truth).query(line)[0].mean() + cKDTree(line).query(truth)[0].mean()) / 2)


def _one(args):
    from dental_ai.cache import cached_preparation
    from dental_ai.eval_crown import _labels, _load
    from dental_ai.geometry import estimate_arch_frame
    from dental_ai.margin.detect import detect_margin, edge_sharpness, resample_loop
    from dental_ai.margin.learned import detect_in_scan
    from dental_ai.synthetic import REAL_PREP_KW

    zip_path, pid, seg_model, out, per_jaw, seed, model = args
    rows = []
    try:
        meshes = {jaw: _load(zip_path, pid, jaw) for jaw in ("upper", "lower")}
        labels = {jaw: _labels(meshes[jaw], pid, jaw, seg_model, Path(out) / "labels") for jaw in meshes}
    except Exception as exc:  # noqa: BLE001
        return [{"patient": pid, "error": str(exc)[:200]}]
    rng = np.random.default_rng(seed)
    for jaw in ("upper", "lower"):
        mesh, lab = meshes[jaw], labels[jaw]
        present = {int(f) for f in np.unique(lab) if f}
        cand = sorted(f for f in present if f % 10 in (4, 5, 6) and f - 1 in present and f + 1 in present)
        try:
            up = estimate_arch_frame(mesh, lab, jaw).occlusal
        except ValueError:
            continue
        for fdi in rng.permutation(cand)[:per_jaw].tolist():
            r = {"patient": pid, "jaw": jaw, "fdi": fdi}
            try:
                prep, truth, _ = cached_preparation(mesh, lab, jaw, fdi, **REAL_PREP_KW)
                sharp = edge_sharpness(prep)
                verts = np.asarray(prep.vertices)
                geo = detect_margin(prep, lab, fdi, up=up, sharpness=sharp)
                lines = {"geo": geo.points}
                r["geo_conf"] = geo.confidence
                try:
                    L = detect_in_scan(model, prep, lab, fdi)
                except ValueError as exc:
                    L = None
                    r["learned_fail"] = str(exc)[:80]
                if L is not None:
                    lines["learned"] = L
                    r["dist_lg"] = _err(L, geo.points)
                    tree = cKDTree(verts)
                    r["sharp_l"] = float(sharp[tree.query(L)[1]].mean())
                    r["sharp_g"] = float(sharp[tree.query(geo.points)[1]].mean())
                    for rad in (0.5, 0.8, 1.2):
                        lines[f"snap{rad}"] = detect_margin(prep, lab, fdi, up=up, sharpness=sharp, initial=L,
                                                            search_radius=rad, max_height_diff=rad).points
                r.update({k: _err(v, truth) for k, v in lines.items()})
            except Exception as exc:  # noqa: BLE001
                r["error"] = f"{type(exc).__name__}: {exc}"[:200]
            rows.append(r)
    return rows


def main():
    import multiprocessing as mp

    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True)
    p.add_argument("--seg-model", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--patients", type=int, default=60)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "labels").mkdir(exist_ok=True)
    for f in Path(a.labels_dir).glob("*.npy"):
        dst = a.out / "labels" / f.name
        if not dst.exists():
            dst.write_bytes(f.read_bytes())
    names = zipfile.ZipFile(a.zip).namelist()
    pids = sorted({n.split("/")[0] for n in names if n.endswith("ios_upper.stl") and n.replace("upper", "lower") in names})
    pids = [pids[i] for i in np.random.default_rng(1).permutation(len(pids))[: a.patients]]
    jobs = [(a.zip, pid, a.seg_model, str(a.out), 1, i, a.model) for i, pid in enumerate(pids)]
    rows = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r in pool.imap_unordered(_one, jobs):
            rows += r
    (a.out / "rows.json").write_text(json.dumps(rows))
    ok = [r for r in rows if "geo" in r]
    print("teeth", len(ok), "learned fail", sum("learned_fail" in r for r in rows), "errors", sum("error" in r for r in rows))

    def pick(name, fn):
        e = np.array([fn(r) for r in ok])
        g = np.array([r["geo"] for r in ok])
        print(f"{name:28s} mean {e.mean():.3f} median {np.median(e):.3f} p90 {np.percentile(e, 90):.3f} "
              f">1mm {(e > 1).mean():.2f} better-than-geo {(e < g - 1e-9).mean():.2f}")

    pick("geo", lambda r: r["geo"])
    pick("learned(else geo)", lambda r: r.get("learned", r["geo"]))
    for rad in (0.5, 0.8, 1.2):
        pick(f"snap{rad}", lambda r, k=f"snap{rad}": r.get(k, r["geo"]))
    for t in (0.4, 0.6, 0.8, 1.0, 1.5):
        pick(f"gate{t}", lambda r, t=t: r["learned"] if "learned" in r and r["dist_lg"] < t else r["geo"])
        pick(f"gate{t}+snap0.5", lambda r, t=t: r["snap0.5"] if "learned" in r and r["dist_lg"] < t else r["geo"])
    pick("sharper", lambda r: r["learned"] if "learned" in r and r["sharp_l"] >= r["sharp_g"] else r["geo"])
    pick("oracle", lambda r: min(r.get("learned", 9), r["geo"]))


if __name__ == "__main__":
    main()
