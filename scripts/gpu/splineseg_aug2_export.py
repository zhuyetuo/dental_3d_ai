"""TeethSplineSeg 第二轮微调的追加数据：每个训练口扫再导出 4 份（11-24）。

真实缺牙里漏检最多的是间隙闭合的前磨牙（正畸拔牙），所以多给闭合缺隙：
  c2 / c3  另两种随机的“缺牙且间隙闭合”
  m2       另一种随机缺牙（1–3 颗，留着缺隙）
  mc       先闭合一处缺隙，再在别处缺 1–2 颗（两种情况同时出现）
和第一轮的 原样 / m1 / c1 一起放进 data/splineseg_aug（同样的目录结构），列表写到 training2_*.txt。

    python splineseg_aug2_export.py [--workers 40]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

R = Path.home() / "zyt/dental_3d_ai"
sys.path.insert(0, str(R / "ml"))
OUT = R / "data/splineseg_aug"


def _save(name, jaw, m, lab):
    d = OUT / jaw / name
    obj = d / f"{name}_{jaw}.obj"
    if not obj.exists():
        d.mkdir(parents=True, exist_ok=True)
        m.export(obj)
        lab = np.asarray(lab).astype(int).tolist()
        (d / f"{name}_{jaw}.json").write_text(json.dumps({"id_patient": name, "jaw": jaw, "labels": lab, "instances": lab}))


def _one(args):
    item, seed = args
    from dental_ai.eval_missing_seg import pick_missing
    from dental_ai.synthetic import pick_closed_gap, simulate_closed_gap, simulate_missing

    case, jaw = item.name.rsplit("_", 1)
    names = []
    try:
        mesh, labels = item.load()
        labels = np.asarray(labels)
        rng = np.random.default_rng(seed)
        out = []
        for tag in ("c2", "c3"):
            t = pick_closed_gap(labels, rng)
            if t:
                out.append((tag, *simulate_closed_gap(mesh, labels, jaw, t, closure=float(rng.uniform(0.7, 1.0)))))
        t = pick_missing(labels, rng)
        if t:
            out.append(("m2", *simulate_missing(mesh, labels, jaw, t)))
        t = pick_closed_gap(labels, rng)
        if t:
            m, lab = simulate_closed_gap(mesh, labels, jaw, t, closure=float(rng.uniform(0.7, 1.0)))
            lab = np.asarray(lab)
            t2 = pick_missing(lab, rng, n=int(rng.choice([1, 2])))
            if t2:
                out.append(("mc", *simulate_missing(m, lab, jaw, t2)))
        for tag, m, lab in out:
            _save(f"{case}{tag}", jaw, m, lab)
            names.append(f"{case}{tag}_{jaw}")
    except Exception as exc:  # noqa: BLE001
        print("skip", item.name, type(exc).__name__, exc, flush=True)
    return jaw, names


def main():
    import multiprocessing as mp

    from dental_ai.datasets import teeth3ds_zip_scans

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    fair = {l.strip() for l in (R / "data/fair_test.txt").read_text().splitlines() if l.strip()}
    items = [s for s in teeth3ds_zip_scans(R / "data/drive/raw/Teeth3DS") if s.name not in fair][: a.limit]
    print(len(items), "个训练口扫（已去掉公平测试集）", flush=True)
    lists = {"lower": [], "upper": []}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, (jaw, names) in enumerate(pool.imap_unordered(_one, [(s, 5000 + k) for k, s in enumerate(items)]), 1):
            lists[jaw] += names
            if i % 200 == 0:
                print(f"[{i}/{len(items)}]", flush=True)
    for jaw in ("lower", "upper"):
        (OUT / f"training2_{jaw}.txt").write_text("\n".join(sorted(lists[jaw])) + "\n")
    print({j: len(v) for j, v in lists.items()}, flush=True)


if __name__ == "__main__":
    main()
