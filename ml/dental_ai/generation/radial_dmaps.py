"""给径向深度图缓存里的每颗牙算一张“学习模型 D 的输出”图，用来训练“在 D 上补细节”的扩散模型（radial_diffusion train --with-d）。

    python -m dental_ai.generation.radial_dmaps --radial ../data/radial/train --zips ../data/drive/raw/Teeth3DS --gen-model ../models/generation/toothgen_open.pt
    python -m dental_ai.generation.radial_dmaps --radial ../data/radial/bite --bite-zip <Bite2Text.zip> --labels ../data/bite_labels --gen-model ...

输出 data/radial_d/<split>/<同名>.npz：fdi（和缓存对齐）、d（D 的牙冠深度图，float16）、has_d（D 生成成功）。
D 自己会把目标牙从上下文里去掉（只看其余牙齿和对颌），所以不用真的把牙挖掉；投影的原点、左右翻转和缓存完全一样，
D 的牙冠和推理时一样先采样表面再 splat、再 clean_depth（见 RadialDiffusionGenerator）。
注意：训练集的牙 D 训练时见过，输出比没见过的牙好；训练扩散模型时用 jitter_d 加扰动。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .radial import SIZE, _run_pool, splat


def d_maps(mesh, labels, jaw, antagonist, fdis, origins, gen) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    from ..detection.missing import MissingTooth
    from .base import GenerationContext
    from .radial_diffusion import clean_depth

    ctx = GenerationContext(mesh=mesh, labels=np.asarray(labels), jaw=jaw, antagonist=antagonist)
    out, ok = np.zeros((len(fdis), SIZE, SIZE), np.float16), np.zeros(len(fdis), bool)
    for k, (fdi, origin) in enumerate(zip(fdis, origins)):
        fdi = int(fdi)
        try:
            tooth = gen.generate(ctx, MissingTooth(fdi, "space", None, None, None, 0.0, ""))
        except (ValueError, LookupError):
            continue
        if tooth.info.get("learned_fallback"):
            continue
        sign = np.array([-1.0, 1.0, 1.0]) if fdi // 10 in (2, 3) else np.ones(3)
        pts = np.asarray(trimesh.sample.sample_surface(tooth.crown, 150000, seed=0)[0])
        d = clean_depth(splat((pts - np.asarray(origin, float)) * sign, np.zeros(3)))
        if (d > 0).sum() < 1500:
            continue
        out[k], ok[k] = d.astype(np.float16), True
    return out, ok


def _gen(model):
    import os

    os.environ["GEN_DEVICE"] = "cpu"
    import torch

    torch.set_num_threads(1)
    from .learned import LearnedGenerator

    return LearnedGenerator(model, fallback=None, device="cpu")


def _one_teeth3ds(args):
    item, cache, out, model = args
    try:
        dest = Path(out) / f"{item.name}.npz"
        if dest.exists():
            return 1
        with np.load(cache) as z:
            fdis, origins = z["fdi"], z["origin"]
        mesh, labels = item.load()
        d, ok = d_maps(mesh, labels, item.name.rsplit("_", 1)[1], None, fdis, origins, _gen(model))
        np.savez_compressed(dest, fdi=fdis, d=d, has_d=ok)
        return int(ok.sum())
    except Exception as exc:  # noqa: BLE001
        print(f"跳过 {item.name}: {type(exc).__name__}: {exc}", flush=True)
        return 0


def _one_bite(args):
    cache, zip_path, labels_dir, seg_model, out, model = args
    try:
        import trimesh

        from ..eval_crown import _labels, _load
        from .radial import _canonical

        cache = Path(cache)
        dest = Path(out) / cache.name
        if dest.exists():
            return 1
        pid, jaw = cache.stem.rsplit("_", 1)
        other = "lower" if jaw == "upper" else "upper"
        with np.load(cache) as z:
            fdis, origins = z["fdi"], z["origin"]
        mesh = _load(zip_path, pid, jaw)
        labels = _labels(mesh, pid, jaw, seg_model, Path(labels_dir))
        canon, rot = _canonical(mesh, labels, jaw)
        o = _load(zip_path, pid, other)
        ant = trimesh.Trimesh(np.asarray(o.vertices) @ rot.T, o.faces, process=False)
        d, ok = d_maps(canon, labels, jaw, ant, fdis, origins, _gen(model))
        np.savez_compressed(dest, fdi=fdis, d=d, has_d=ok)
        return int(ok.sum())
    except Exception as exc:  # noqa: BLE001
        print(f"跳过 {cache}: {type(exc).__name__}: {exc}", flush=True)
        return 0


def main() -> None:
    p = argparse.ArgumentParser(description="径向深度图缓存里每颗牙的 D 输出图")
    p.add_argument("--radial", type=Path, required=True, help="data/radial/<split>")
    p.add_argument("--gen-model", required=True)
    p.add_argument("--zips", type=Path, help="Teeth3DS zip 目录（Teeth3DS 缓存）")
    p.add_argument("--bite-zip", help="Bite2Text.zip（Bite2Text 缓存）")
    p.add_argument("--labels", default="../data/bite_labels")
    p.add_argument("--seg-model", default="../models/segmentation/deep.pt")
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    from .radial_diffusion import d_dir_for

    out = d_dir_for(args.radial)
    out.mkdir(parents=True, exist_ok=True)
    caches = sorted(args.radial.glob("*.npz"))[: args.limit]
    model = str(Path(args.gen_model).resolve())
    if args.zips:
        from ..datasets import teeth3ds_zip_scans

        items = {s.name: s for s in teeth3ds_zip_scans(args.zips)}
        jobs = [(items[c.stem], str(c), str(out), model) for c in caches if c.stem in items]
        print(f"{len(jobs)} 个颌（Teeth3DS）", flush=True)
        print("完成，D 成功的牙", _run_pool(_one_teeth3ds, jobs, args.workers), flush=True)
    else:
        jobs = [(str(c), args.bite_zip, args.labels, args.seg_model, str(out), model) for c in caches]
        print(f"{len(jobs)} 个颌（Bite2Text）", flush=True)
        print("完成，D 成功的牙", _run_pool(_one_bite, jobs, args.workers), flush=True)


if __name__ == "__main__":
    main()
