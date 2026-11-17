"""只给“末端牙”（只有一侧有邻牙）补径向深度图样本，用新的外推原点。输出 data/radial/<split>_term，格式和原来的缓存一样。
用法：RADIAL_ONE_SIDE=1 python scripts/data/radial_terminal_cache.py train|bite|bite_val（要打开外推原点）"""
import json
import sys
from pathlib import Path

R = Path.home() / "zyt/dental_3d_ai"
sys.path.insert(0, str(R / "ml"))
from dental_ai.generation import radial  # noqa: E402
from dental_ai.generation import radial_gpu as rg  # noqa: E402
from dental_ai.generation.learned import arch_index  # noqa: E402

_orig = radial.gap_origin


def terminal_only(centroids, target):
    ti = arch_index(target, target)
    idx = {f: arch_index(f, target) for f in centroids if f != target and (f // 10 in (1, 2)) == (target // 10 in (1, 2))}
    if any(i > ti for i in idx.values()) and any(i < ti for i in idx.values()):
        return None  # 两侧都有：原来的缓存里已经有了
    return _orig(centroids, target)


radial.gap_origin = terminal_only
rg.MIN_TEETH = 1


def make_jobs(split):
    only = {f.stem for f in (R / "data/radial" / split).glob("*.npz")}
    out = R / "data/radial" / f"{split}_term"
    if split == "train":
        from dental_ai.datasets import teeth3ds_zip_scans

        Z = R / "data/drive/raw/Teeth3DS"
        S = Z / "Teeth3DS+/Teeth3DS_train_test_split"
        items = [s for sp in (S / "training_lower.txt", S / "training_upper.txt") for s in teeth3ds_zip_scans(Z, sp)]
        return [s for s in items if s.name in only], rg._load_teeth3ds_raw, out
    import zipfile

    bz = str(Path.home() / "zyt/dental_data/Bite2Text.zip")
    names = set(zipfile.ZipFile(bz).namelist())
    pids = sorted({n.split("/")[0] for n in names if n.endswith("ios_upper.stl") and n.replace("upper", "lower") in names})
    pids = [q for q in pids if f"{q}_upper" in only or f"{q}_lower" in only]
    return [(bz, q, str(R / "data/bite_labels"), str(R / "models/segmentation/deep.pt")) for q in pids], rg._load_bite_raw, out


if __name__ == "__main__":
    split = sys.argv[1]
    jobs, loader, out = make_jobs(split)
    print(json.dumps({"split": split, **rg.cache_gpu(jobs, loader, out, 16)}, ensure_ascii=False), flush=True)
