"""分割训练数据加“模拟预备体”变体：分割模型没见过磨过的牙，真实备牙口扫上预备体常分错（11-23 在 Cegep 上看到），
颈缘线检测从分割边界出发，跟着错。这里把训练扫描里随机 1–3 颗牙按临床磨除量磨成预备体（synthetic.simulate_preparation，
参数随机：肩台高度、轴面 / 咬合面磨除量、聚合角、邻面是渐变还是台阶），标签不变（预备体仍是那颗牙），
写成和 deep.py cache 一样格式的 <扫描>__p<k>.npz（同一扫描的变体在验证集同一边）。

    python -m dental_ai.segmentation.prep_aug --zips ../data/drive/raw/Teeth3DS --split <training_lower.txt> <training_upper.txt> \
        --extra-dir ../data/poseidon3d/train --variants 1 --out ../data/deep_cache_prepsim
    # 训练用的目录 = 原来的缓存 + 这些变体（软链接），stats.json 沿用原来的
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .deep import CACHE_POINTS


def _one(args):
    scan, out_dir, seed, variant = args
    dest = Path(out_dir) / f"{scan.name}__p{variant}.npz"
    if dest.exists():
        return scan.name
    try:
        from ..synthetic import simulate_preparation
        from .baseline import features, fdi_to_class

        if hasattr(scan, "load"):
            mesh, labels = scan.load()
        else:
            from ..io import load_labels, load_mesh

            mesh, labels = load_mesh(scan.mesh_path), load_labels(scan.gt_path)[0]
        labels = np.asarray(labels)
        rng = np.random.default_rng(seed)
        present = sorted(int(f) for f in np.unique(labels) if f > 0 and f % 10 != 8)
        if len(present) < 4:
            return None
        # 后牙更常做牙冠：前磨牙、磨牙权重 3，前牙 1
        w = np.array([3.0 if f % 10 >= 4 else 1.0 for f in present])
        teeth = rng.choice(present, size=min(len(present), int(rng.integers(1, 4))), replace=False, p=w / w.sum())
        done = 0
        for fdi in teeth:
            try:
                mesh, _ = simulate_preparation(
                    mesh, labels, scan.jaw, int(fdi),
                    margin_above_gum=float(rng.uniform(0.2, 1.2)), axial=float(rng.uniform(0.8, 1.6)), occlusal=float(rng.uniform(1.0, 2.2)),
                    taper_deg=float(rng.uniform(4, 12)), scallop=float(rng.uniform(0.3, 0.6)), proximal_band=float(rng.choice([0.2, 0.4, 1.0])))
                done += 1
            except Exception:  # noqa: BLE001  个别牙磨不了（太小、标签碎）跳过
                continue
        if not done:
            return None
        x = features(mesh, scan.jaw)
        y = fdi_to_class(labels, scan.jaw)
        referenced = np.unique(mesh.faces)
        pick = rng.choice(referenced, min(CACHE_POINTS, len(referenced)), replace=False)
        np.savez_compressed(dest, x=x[pick].astype(np.float16), y=y[pick].astype(np.uint8))
        return scan.name
    except Exception as exc:  # noqa: BLE001
        print(f"跳过 {scan.name}: {exc}", flush=True)
        return None


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    from ..datasets import teeth3ds_zip_scans
    from ..evaluation import find_scans

    p = argparse.ArgumentParser(description="分割训练数据：模拟预备体变体")
    p.add_argument("--zips", type=Path)
    p.add_argument("--split", type=Path, nargs="*", default=[])
    p.add_argument("--extra-dir", type=Path, nargs="*", default=[])
    p.add_argument("--variants", type=int, default=1)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    scans = [s for f in a.split for s in teeth3ds_zip_scans(a.zips, f)] if a.zips else []
    for d in a.extra_dir:
        scans += find_scans(d, d)
    scans = scans[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, a.out, 9000011 * v + i, v) for v in range(1, a.variants + 1) for i, s in enumerate(scans)]
    print(f"{len(scans)} 个扫描，{len(jobs)} 个预备体变体", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        done = [n for n in pool.map(_one, jobs, chunksize=2) if n]
    print(f"完成 {len(done)} 个", flush=True)


if __name__ == "__main__":
    main()
