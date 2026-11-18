"""Bite2Text 口扫一次性解析好存到硬盘（data/bite2text_mesh/<病人>_<颌>.npz：verts、faces），评分卡反复读口扫时直接读它
（Linux 会把常用的文件留在内存里，读起来和内存差不多快）。比每次从 19 GB 的 zip 现解压、现解析快 40 多倍（17 ms vs 750 ms）。

    python scripts/data/shm_cache.py [进程数]
"""
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

Z = str(Path.home() / "zyt/dental_data/Bite2Text.zip")
OUT = Path.home() / "zyt/dental_3d_ai/data/bite2text_mesh"


def one(name):
    import trimesh

    pid, jaw = name.split("/")[0], "upper" if name.endswith("upper.stl") else "lower"
    dst = OUT / f"{pid}_{jaw}.npz"
    if dst.exists():
        return 0
    try:
        mesh = trimesh.load(io.BytesIO(zipfile.ZipFile(Z).read(name)), file_type="stl")
        if isinstance(mesh, trimesh.Scene):  # 个别文件读出来是场景（多块）：合成一个网格，和 eval_crown 读 zip 时的处理一致前先跳过
            return -1
        mesh.remove_unreferenced_vertices()
        tmp = dst.with_name(dst.stem + ".tmp.npz")
        np.savez(tmp, verts=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces, np.int32))
        tmp.rename(dst)
        return 1
    except Exception:  # noqa: BLE001
        return -1


if __name__ == "__main__":
    import multiprocessing as mp
    import time

    OUT.mkdir(parents=True, exist_ok=True)
    names = [n for n in zipfile.ZipFile(Z).namelist() if n.endswith(("ios_upper.stl", "ios_lower.stl"))]
    t0 = time.time()
    with mp.get_context("spawn").Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 24) as pool:
        res = list(pool.imap_unordered(one, names, chunksize=4))
    print(f"新存 {res.count(1)} 个，跳过（场景 / 读不了，评分卡照旧读 zip）{res.count(-1)} 个，共 {len(names)} 个，{time.time() - t0:.0f} 秒")
