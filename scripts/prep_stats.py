"""真实预备体（FDI 16 Crown Dataset：预备体 + 技师牙冠）和模拟预备体（synthetic.simulate_preparation：预备体 + 原来的牙）
用同一套量法比：各高度的磨除量（近远中 / 颊舌）、咬合面磨除量、聚合角、预备体高度、颈缘线起伏。

    python /tmp/prep_stats.py [--sim-kw '{"axial":1.5}'] [--n-real 1500] [--patients 60]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
FR = (0.1, 0.3, 0.5, 0.7, 0.9)


def measure(prep, margin, crown, up, md):
    up = up / np.linalg.norm(up)
    md = md - (md @ up) * up
    md /= np.linalg.norm(md)
    bl = np.cross(up, md)
    T = np.stack([md, up, bl])
    P, M, C = prep @ T.T, margin @ T.T, crown @ T.T
    ym = float(np.median(M[:, 1]))
    top = float(P[:, 1].max())
    H = top - ym
    c = (P[:, [0, 2]].min(0) + P[:, [0, 2]].max(0)) / 2
    r = {"H": H, "scallop": float(np.percentile(M[:, 1], 95) - np.percentile(M[:, 1], 5))}
    wid = {}
    for f in FR:
        y = ym + f * H
        bp, bc = P[np.abs(P[:, 1] - y) < 0.2], C[np.abs(C[:, 1] - y) < 0.2]
        if len(bp) < 10 or len(bc) < 10:
            continue
        for k, ax in (("md", 0), ("bl", 2)):
            wp = bp[:, ax].max() - bp[:, ax].min()
            wc = bc[:, ax].max() - bc[:, ax].min()
            wid[(k, f)] = wp
            r[f"red_{k}_{int(f * 100)}"] = float((wc - wp) / 2)
    for k in ("md", "bl"):
        if (k, 0.3) in wid and (k, 0.7) in wid:
            r[f"taper_{k}"] = float(np.degrees(2 * np.arctan((wid[(k, 0.3)] - wid[(k, 0.7)]) / 2 / (0.4 * H))))
    # 咬合面：0.5 mm 网格上各自的最高点，取预备体上部区域（高度 > 60%）的格子，牙冠减预备体
    def hmap(Q):
        ij = np.floor((Q[:, [0, 2]] - c) / 0.5).astype(int)
        d = {}
        for (i, j), y in zip(map(tuple, ij), Q[:, 1]):
            if y > d.get((i, j), -1e9):
                d[(i, j)] = y
        return d
    hp, hc = hmap(P), hmap(C)
    diffs = [hc[k] - v for k, v in hp.items() if k in hc and v > ym + 0.6 * H]
    r["occl_red"] = float(np.median(diffs)) if diffs else np.nan
    r["occl_red_top"] = float(C[:, 1].max() - top)
    return r


def real_case(d):
    import trimesh

    from dental_ai.eval_crownfusion import load_case, margin_loop

    try:
        pts, props, prep, crown = load_case(d)
        p = trimesh.sample.sample_surface(prep, 20000, seed=0)[0]
        c = trimesh.sample.sample_surface(crown, 20000, seed=0)[0]
        return measure(p, margin_loop(prep), c, np.array([0, 1.0, 0]), np.array([1.0, 0, 0]))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def sim_patient(args):
    import trimesh

    from dental_ai.eval_crown import _load
    from dental_ai.geometry import estimate_arch_frame
    from dental_ai.synthetic import simulate_preparation

    zp, pid, labdir, kw = args
    out = []
    for jaw in ("upper", "lower"):
        try:
            mesh = _load(zp, pid, jaw)
            f = Path(labdir) / f"{pid}_{jaw}.npy"
            if not f.exists():
                continue
            lab = np.load(f).astype(int)
            present = {int(x) for x in np.unique(lab) if x}
            up = estimate_arch_frame(mesh, lab, jaw).occlusal
        except Exception:  # noqa: BLE001
            continue
        for fdi in sorted(x for x in present if x % 10 in (4, 5, 6) and x - 1 in present and x + 1 in present):
            try:
                prep, truth = simulate_preparation(mesh, lab, jaw, fdi, **kw)
                v0, v1 = np.asarray(mesh.vertices), np.asarray(prep.vertices)
                moved = np.linalg.norm(v1 - v0, axis=1) > 1e-6
                t = lab == fdi
                hm = float(np.median(truth @ up))
                nat = v0[t & (v0 @ up > hm)]
                pp = np.concatenate([v1[moved], truth])
                md = v0[lab == fdi + 1].mean(0) - v0[lab == fdi - 1].mean(0)
                r = measure(pp, truth, nat, up, md)
                r.update({"fdi": fdi, "pid": pid})
                out.append(r)
            except Exception as exc:  # noqa: BLE001
                out.append({"error": str(exc)[:100]})
    return out


def summarize(rows, title):
    rows = [r for r in rows if "error" not in r]
    keys = ["H", "scallop", "occl_red", "occl_red_top", "taper_md", "taper_bl"] + [f"red_{k}_{int(f * 100)}" for k in ("md", "bl") for f in FR]
    print(f"== {title}  n={len(rows)}")
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])])
        if len(v):
            out[k] = (round(float(np.median(v)), 2), round(float(np.percentile(v, 25)), 2), round(float(np.percentile(v, 75)), 2))
            print(f"  {k:14s} 中位 {out[k][0]:6.2f}  四分位 [{out[k][1]:6.2f} {out[k][2]:6.2f}]")
    return out


def main():
    import multiprocessing as mp
    import zipfile

    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-kw", default="{}")
    ap.add_argument("--n-real", type=int, default=1500)
    ap.add_argument("--patients", type=int, default=60)
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--out", default="/tmp/prep_stats.json")
    a = ap.parse_args()
    res = {}
    with mp.get_context("spawn").Pool(32) as pool:
        if not a.skip_real:
            D = Path("/home/zyt/zyt/dental_data/crownfusion/CrownFusion")
            ids = json.load(open(D / "training.json"))[: a.n_real]
            res["real"] = summarize(pool.map(real_case, [D / "train" / str(i) for i in ids], chunksize=8), "真实预备体（16，技师牙冠当原牙）")
        zp = "/home/zyt/zyt/dental_data/Bite2Text.zip"
        names = zipfile.ZipFile(zp).namelist()
        pids = sorted({n.split("/")[0] for n in names if n.endswith("ios_upper.stl") and n.replace("upper", "lower") in names})
        pids = [pids[i] for i in np.random.default_rng(1).permutation(len(pids))[: a.patients]]
        kw = json.loads(a.sim_kw)
        rows = [r for rr in pool.map(sim_patient, [(zp, p, "/home/zyt/zyt/dental_3d_ai/runs/crown_quality/labels", kw) for p in pids]) for r in rr]
        ok = [r for r in rows if "error" not in r]
        res["sim_molar"] = summarize([r for r in ok if r["fdi"] % 10 == 6], f"模拟预备体 磨牙 {kw}")
        res["sim_upper_molar"] = summarize([r for r in ok if r["fdi"] in (16, 26)], f"模拟预备体 上颌第一磨牙 {kw}")
        res["sim_premolar"] = summarize([r for r in ok if r["fdi"] % 10 in (4, 5)], f"模拟预备体 前磨牙 {kw}")
        print("errors", len(rows) - len(ok), [r["error"] for r in rows if "error" in r][:3])
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
