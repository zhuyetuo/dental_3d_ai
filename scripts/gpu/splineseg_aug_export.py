"""给 TeethSplineSeg 微调准备数据：Teeth3DS（去掉公平测试集）每个口扫导出 原样 / 模拟缺牙 / 缺隙闭合 三份，
Teeth3DS 目录结构（<颌>/<病例>/<病例>_<颌>.obj + .json），training_*.txt 列全部，testing_*.txt 是公平测试集（软链接）。

    python ss_aug_export.py [--workers 40]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

R = Path.home() / "zyt/dental_3d_ai"
sys.path.insert(0, str(R / "ml"))
OUT = R / "data/splineseg_aug"


def _one(args):
    item, seed = args
    from dental_ai.eval_missing_seg import pick_missing
    from dental_ai.synthetic import pick_closed_gap, simulate_closed_gap, simulate_missing

    case, jaw = item.name.rsplit("_", 1)
    names = []
    try:
        mesh, labels = item.load()
        labels = np.asarray(labels)
        variants = [("", mesh, labels)]
        rng = np.random.default_rng(seed)
        t = pick_missing(labels, rng)
        if t:
            variants.append(("m1", *simulate_missing(mesh, labels, jaw, t)))
        t = pick_closed_gap(labels, rng)
        if t:
            variants.append(("c1", *simulate_closed_gap(mesh, labels, jaw, t, closure=float(rng.uniform(0.8, 1.0)))))
        for tag, m, lab in variants:
            name = f"{case}{tag}"
            d = OUT / jaw / name
            obj = d / f"{name}_{jaw}.obj"
            if not obj.exists():
                d.mkdir(parents=True, exist_ok=True)
                m.export(obj)
                lab = np.asarray(lab).astype(int).tolist()
                (d / f"{name}_{jaw}.json").write_text(json.dumps({"id_patient": name, "jaw": jaw, "labels": lab,
                                                                    "instances": lab}))
            names.append(f"{name}_{jaw}")
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
    OUT.mkdir(parents=True, exist_ok=True)
    lists = {"lower": [], "upper": []}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i, (jaw, names) in enumerate(pool.imap_unordered(_one, [(s, 1000 + k) for k, s in enumerate(items)]), 1):
            lists[jaw] += names
            if i % 200 == 0:
                print(f"[{i}/{len(items)}]", flush=True)
    # 公平测试集：链接现成的 OBJ / JSON，作为 testing
    fair_root = R / "data/splineseg_root"
    test = {"lower": [], "upper": []}
    for jaw in ("lower", "upper"):
        for obj in sorted((fair_root / jaw).glob("*/*.obj")):
            dest = OUT / jaw / obj.parent.name
            if not dest.exists():
                dest.symlink_to(obj.parent.resolve())
            test[jaw].append(obj.stem)
        (OUT / f"training_{jaw}.txt").write_text("\n".join(sorted(lists[jaw])) + "\n")
        (OUT / f"testing_{jaw}.txt").write_text("\n".join(sorted(test[jaw])) + "\n")
    print({j: (len(lists[j]), len(test[j])) for j in lists}, flush=True)


if __name__ == "__main__":
    main()
