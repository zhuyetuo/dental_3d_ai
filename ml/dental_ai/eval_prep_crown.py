"""预备体牙冠评分卡（模拟预备体）：在真实口扫（Bite2Text）上把前磨牙 / 磨牙按临床磨除量磨成预备体（synthetic.simulate_preparation：
轴面 1 mm、咬合面 1.5 mm、6° 聚合角、肩台在牙颈线上 0.8 mm），颈缘线真值和原来的牙都已知。跑流程生成预备体牙冠，量：
- 颈缘线：检测出来的离真值多远（平均 / 95%，mm）；
- 外形：牙冠外表面和原来的牙差多少（和缺失牙评分卡同一套 ref_* 指标，只比颈缘线以上的部分）；
- 贴合：预备体（磨过的面）到牙冠的距离——就是粘接间隙，应该接近设置值；穿进预备体的比例；
- 封闭、邻接、咬合（score_crown）。
每颗牙跑两遍：自动检测颈缘线（auto）、直接给真值颈缘线（truth，只看牙冠本身）。

    python -m dental_ai.eval_prep_crown --zip <Bite2Text.zip> --seg-model ../models/segmentation/deep.pt --gen-model ../models/generation/toothgen_open.pt \
        --patients 40 --out ../runs/prep_card
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .eval_crown import _labels, _load


def _one(args) -> list[dict]:
    from .crown_quality import _crop, score_crown
    from .eval_generation import _Fixed
    from .generation.contacts import signed_distance_to_surface
    from .generation.learned import make_generator
    from .geometry import estimate_arch_frame, tooth_mesh
    from .pipeline import run
    from .synthetic import simulate_preparation

    zip_path, pid, seg_model, gen_model, out, per_jaw, seed = args
    rows: list[dict] = []
    try:
        meshes = {jaw: _load(zip_path, pid, jaw) for jaw in ("upper", "lower")}
        labels = {jaw: _labels(meshes[jaw], pid, jaw, seg_model, Path(out) / "labels") for jaw in meshes}
    except Exception as exc:  # noqa: BLE001
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
                prep, truth = simulate_preparation(mesh, lab, jaw, fdi)
                # 真值颈缘线是一堆没排序的顶点（肩台外缘 + 邻面刃状边缘），生成牙冠要的是首尾相连的一圈：按绕牙轴的角度排好
                c = truth.mean(axis=0)
                ax = np.cross(up, [1.0, 0.0, 0.0])
                ax /= np.linalg.norm(ax)
                ay = np.cross(up, ax)
                truth = truth[np.argsort(np.arctan2((truth - c) @ ay, (truth - c) @ ax))]
                moved = np.linalg.norm(np.asarray(prep.vertices) - np.asarray(mesh.vertices), axis=1) > 1e-6
                prep_pts = np.asarray(prep.vertices)[moved]  # 磨过的面（颈缘线以上的预备体）
                # 原来的牙只取颈缘线以上：外形只比这部分
                rv = np.asarray(real.vertices)
                hm = float(np.median(truth @ up))
                real_top = real.submesh([np.flatnonzero((rv[real.faces] @ up).mean(axis=1) > hm)], append=True)
                for mode in ("auto", "truth"):
                    with tempfile.TemporaryDirectory() as tmp:
                        gen = make_generator(gen_model)
                        overrides = {fdi: {"points": truth.tolist()}} if mode == "truth" else None
                        report = run(prep, jaw, _Fixed(lab), tmp, generator=gen, antagonist=ant, prepared_teeth=[fdi], generate_teeth=[],
                                     margin_overrides=overrides)
                        done = {g["fdi"]: g for g in report["generated"] if g.get("method") == "prep_crown"}
                        if fdi not in done:
                            rows.append({**base, "kind": mode, "error": "未生成：" + "; ".join(w for w in report["warnings"] if str(fdi) in w)[:160]})
                            continue
                        crown = trimesh.load(Path(tmp) / done[fdi]["file"])
                        mpts = np.asarray(json.loads((Path(tmp) / "margins.json").read_text())[str(fdi)]["points"])
                    # 颈缘线误差（双向平均）
                    d1 = cKDTree(truth).query(mpts)[0]
                    d2 = cKDTree(mpts).query(truth)[0]
                    # 贴合：磨过的预备体面到牙冠的有符号距离。牙冠是封闭实体，组织面的法向朝外（指向空腔），
                    # 预备体的点在空腔里为正 = 粘接间隙；负的是预备体戳进了牙冠材料里（组织面没贴住、倒凹）
                    sd, _ = signed_distance_to_surface(prep_pts, crown, cutoff=2.0)
                    sd = sd[np.isfinite(sd)]
                    extra = {"margin_mean": float((d1.mean() + d2.mean()) / 2), "margin_p95": float(np.percentile(np.concatenate([d1, d2]), 95)),
                             "gap_median": float(np.median(sd)) if len(sd) else None, "gap_p95": float(np.percentile(sd, 95)) if len(sd) else None,
                             "into_prep_rate": float((sd < -0.02).mean()) if len(sd) else None,
                             "source": done[fdi].get("source"), "min_thickness": done[fdi].get("min_thickness_mm")}
                    rows.append({**base, "kind": mode, **extra, **score_crown(crown, up, mesial, distal, ant, reference=real_top)})
            except Exception as exc:  # noqa: BLE001
                rows.append({**base, "kind": "error", "error": f"{type(exc).__name__}: {exc}"[:200]})
    return rows


KEYS = ("margin_mean", "margin_p95", "gap_median", "gap_p95", "into_prep_rate", "ref_chamfer_l1", "ref_rmse", "ref_cusp_err", "ref_fossa_err",
        "mesial_gap", "distal_gap", "occl_clearance", "height_vs_neighbors", "fossa_depth")


def summarize(rows):
    out = {"errors": sum("error" in r for r in rows)}
    for kind in ("auto", "truth"):
        sub = [r for r in rows if r.get("kind") == kind and "error" not in r]
        block = {"n": len(sub), "watertight": round(float(np.mean([bool(r.get("watertight")) for r in sub])), 3) if sub else None}
        for k in KEYS:
            v = np.array([r[k] for r in sub if r.get(k) is not None], float)
            v = v[np.isfinite(v)]
            if len(v):
                block[k] = {"median": round(float(np.median(v)), 3), "p5": round(float(np.percentile(v, 5)), 3), "p95": round(float(np.percentile(v, 95)), 3)}
        out[kind] = block
    return out


def main() -> None:
    import multiprocessing as mp
    import zipfile

    p = argparse.ArgumentParser(description="预备体牙冠评分卡（模拟预备体）")
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--seg-model", required=True)
    p.add_argument("--gen-model", required=True)
    p.add_argument("--patients", type=int, default=40)
    p.add_argument("--per-jaw", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--labels-from", type=Path, help="已有的分割标签缓存目录（复用缺失牙评分卡的）")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(exist_ok=True)
    if args.labels_from:
        for f in args.labels_from.glob("*.npy"):
            dst = args.out / "labels" / f.name
            if not dst.exists():
                dst.write_bytes(f.read_bytes())
    names = zipfile.ZipFile(args.zip).namelist()
    pids = sorted({n.split("/")[0] for n in names if n.endswith("ios_upper.stl") and n.replace("upper", "lower") in names})
    pids = [pids[i] for i in np.random.default_rng(1).permutation(len(pids))[: args.patients]]
    jobs = [(str(args.zip), pid, args.seg_model, args.gen_model, str(args.out), args.per_jaw, i) for i, pid in enumerate(pids)]
    rows = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, jobs)):
            rows += r
            if (i + 1) % 5 == 0:
                print(f"[{i + 1}/{len(jobs)}]", flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(args.out / "crowns.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    s = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
    print(json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
