"""径向深度图的 GPU 版：表面采样、投影（splat）、补洞都是显卡上的张量运算，一个颌的所有牙一起算。

CPU 版（radial.py 的 splat / tooth_maps / scan_maps）逐颗牙用 numpy 和 trimesh 做，数据多了要一两个小时。
这里的约定和 CPU 版完全一样（同样的投影、R_CLIP、补洞规则），结果只差采样的随机性。
CPU 只负责读文件和解析网格（多进程），算的部分都在 GPU 上：

    python -m dental_ai.generation.radial_gpu bench --radial ../data/radial/bite_val --bite-zip <Bite2Text.zip> --limit 20
"""

from __future__ import annotations

import math

import numpy as np

from .radial import CENTER_DROP, MAX_POLAR, R_CLIP, SIZE  # noqa: F401


def sample_surface(verts, faces, n: int, generator=None):
    """按面积均匀采样三角网格表面。verts (V, 3)、faces (F, 3) 都是 GPU 张量；返回 (n, 3)。"""
    import torch

    tri = verts[faces]  # (F, 3, 3)
    area = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).norm(dim=1)
    if len(area) == 0 or float(area.sum()) <= 0:
        return verts.new_zeros((0, 3))
    idx = torch.multinomial(area, n, replacement=True, generator=generator)
    u = torch.rand(n, 2, device=verts.device, generator=generator)
    flip = u.sum(1) > 1
    u[flip] = 1 - u[flip]
    t = tri[idx]
    return t[:, 0] + u[:, :1] * (t[:, 1] - t[:, 0]) + u[:, 1:] * (t[:, 2] - t[:, 0])


def _dilate_cross(m, it: int):
    import torch
    import torch.nn.functional as F

    k = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32, device=m.device).view(1, 1, 3, 3)
    x = m.float()[:, None]
    for _ in range(it):
        x = (F.conv2d(x, k, padding=1) > 0).float()
    return x[:, 0] > 0


def _erode_cross(m, it: int):
    """十字结构腐蚀，图外当作 0（和 scipy.ndimage.binary_erosion 默认 border_value=0 一致）。"""
    import torch
    import torch.nn.functional as F

    k = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.float32, device=m.device).view(1, 1, 3, 3)
    x = (~m).float()[:, None]
    for _ in range(it):
        x = (F.conv2d(F.pad(x, (1, 1, 1, 1), value=1.0), k) > 0).float()
    return ~(x[:, 0] > 0)


def splat(points, batch, n_batch: int, fill: int = 2):
    """points (N, 3)：已经减去投影中心（并左右翻转）的点；batch (N,)：每个点属于哪张图。返回 (n_batch, S, S)，没有点的像素是 0。
    和 radial.splat 一样：每个像素取最近的点的距离；只补被有效像素包围的小洞（binary_closing 十字结构 fill 次 + 3×3 最小值传播）。"""
    import torch
    import torch.nn.functional as F

    r = points.norm(dim=1)
    keep = (r > 1e-6) & (r < R_CLIP)
    p, r, b = points[keep], r[keep], batch[keep]
    theta = torch.arccos((p[:, 2] / r).clamp(-1, 1))
    phi = torch.atan2(p[:, 1], p[:, 0])
    rho = theta / MAX_POLAR
    inside = rho < 1.0
    p, r, b, rho, phi = p[inside], r[inside], b[inside], rho[inside], phi[inside]
    i = torch.floor((rho * torch.cos(phi) * 0.5 + 0.5) * SIZE).long().clamp(0, SIZE - 1)
    j = torch.floor((rho * torch.sin(phi) * 0.5 + 0.5) * SIZE).long().clamp(0, SIZE - 1)
    out = torch.full((n_batch * SIZE * SIZE,), math.inf, device=points.device)
    out.scatter_reduce_(0, b * SIZE * SIZE + i * SIZE + j, r, reduce="amin")
    out = out.view(n_batch, SIZE, SIZE)
    valid = torch.isfinite(out)
    if fill:
        holes = _erode_cross(_dilate_cross(valid, fill), fill) & ~valid
        for _ in range(fill):
            near = -F.max_pool2d(-out[:, None].nan_to_num(posinf=1e9), 3, stride=1, padding=1)[:, 0]
            near = torch.where(near >= 1e9, torch.full_like(near, math.inf), near)
            out = torch.where(holes & ~torch.isfinite(out), near, out)
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def tooth_maps_batch(verts, faces, lab3, targets, origins, flips, antagonist=None, seed: int = 0,
                     n_tooth: int = 150000, n_ctx: int = 300000, n_ant: int = 300000):
    """一个颌、多颗目标牙一次算完。verts/faces 是 GPU 张量，lab3 (F, 3) 是每个面三个顶点的牙号（选面规则和 radial.tooth_maps 一样）；
    targets、origins (T, 3)、flips (T,) 每颗目标牙一行。返回 dict：tooth / ctx / ant 各 (T, S, S)，has_ant (T,)。"""
    import torch

    dev = verts.device
    g = torch.Generator(device=dev).manual_seed(seed)
    T = len(targets)
    origins = torch.as_tensor(np.asarray(origins), dtype=torch.float32, device=dev)
    sign = torch.ones(T, 3, device=dev)
    sign[torch.as_tensor(np.asarray(flips), device=dev).bool(), 0] = -1.0
    tooth_pts, ctx_pts, tb, cb = [], [], [], []
    for k, fdi in enumerate(targets):
        near = (verts[faces[:, 0]] - origins[k]).norm(dim=1) < R_CLIP + 2
        is_t = (lab3 == int(fdi)).all(dim=1)
        is_c = ((lab3 > 0) & (lab3 != int(fdi))).all(dim=1)
        for sel, n, pts_list, b_list in ((is_t & near, n_tooth, tooth_pts, tb), (is_c & near, n_ctx, ctx_pts, cb)):
            pts = sample_surface(verts, faces[sel], n, g) if bool(sel.any()) else verts.new_zeros((0, 3))
            pts_list.append((pts - origins[k]) * sign[k])
            b_list.append(torch.full((len(pts),), k, device=dev, dtype=torch.long))
    out = {"tooth": splat(torch.cat(tooth_pts), torch.cat(tb), T), "ctx": splat(torch.cat(ctx_pts), torch.cat(cb), T)}
    has = torch.zeros(T, dtype=torch.bool, device=dev)
    ant_maps = torch.zeros(T, SIZE, SIZE, device=dev)
    if antagonist is not None:
        av, af = antagonist
        ap, ab = [], []
        for k in range(T):
            near = (av[af[:, 0]] - origins[k]).norm(dim=1) < R_CLIP + 2
            if int(near.sum()) <= 50:
                continue
            pts = (sample_surface(av, af[near], n_ant, g) - origins[k]) * sign[k]
            pts = pts[pts[:, 2] > -1.0]
            ap.append(pts)
            ab.append(torch.full((len(pts),), k, device=dev, dtype=torch.long))
            has[k] = True
        if ap:
            ant_maps = splat(torch.cat(ap), torch.cat(ab), T)
    out["ant"], out["has_ant"] = ant_maps, has
    return out


def _load_teeth3ds(args):
    """CPU 进程里做：读 Teeth3DS 的一个颌（本来就是标准坐标）。"""
    item, cache = args
    try:
        mesh, labels = item.load()
        z = np.load(cache)
        return item.name, item.name.rsplit("_", 1)[1], np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces), \
            np.asarray(labels), None, None, z["fdi"], z["origin"]
    except Exception as exc:  # noqa: BLE001
        return item.name, f"{type(exc).__name__}: {exc}", None, None, None, None, None, None, None


def _load_bite(args):
    """CPU 进程里做：读 Bite2Text 的一个颌和对颌、自动标注（有缓存）、转到标准坐标。"""
    cache, zip_path, labels_dir, seg_model = args
    try:
        from pathlib import Path

        from ..eval_crown import _labels, _load
        from .radial import _canonical

        cache = Path(cache)
        pid, jaw = cache.stem.rsplit("_", 1)
        other = "lower" if jaw == "upper" else "upper"
        mesh = _load(zip_path, pid, jaw)
        labels = _labels(mesh, pid, jaw, seg_model, Path(labels_dir))
        canon, rot = _canonical(mesh, labels, jaw)
        o = _load(zip_path, pid, other)
        z = np.load(cache)
        return cache.stem, jaw, np.asarray(canon.vertices, np.float32), np.asarray(canon.faces), np.asarray(labels), \
            (np.asarray(o.vertices) @ rot.T).astype(np.float32), np.asarray(o.faces), z["fdi"], z["origin"]
    except Exception as exc:  # noqa: BLE001
        return Path(cache).stem, f"{type(exc).__name__}: {exc}", None, None, None, None, None, None, None


def scan_maps_gpu(verts, faces, labels, antagonist=None, seed: int = 0, device: str = "cuda") -> dict | None:
    """radial.scan_maps 的 GPU 版：一个颌每颗牙（不含第三磨牙）挖掉后的三张图，一次算完。返回要存盘的数组（格式同 CPU 版）。"""
    import torch

    from .radial import gap_origin

    labels = np.asarray(labels)
    centroids = {int(f): verts[labels == f].mean(axis=0) for f in np.unique(labels) if f > 0 and (labels == f).sum() >= 200}
    fdis, origins = [], []
    for fdi in sorted(centroids):
        if fdi % 10 > 7:
            continue
        o = gap_origin(centroids, fdi)
        if o is not None:
            fdis.append(fdi)
            origins.append(o)
    if len(fdis) < 4:
        return None
    V = torch.as_tensor(verts, dtype=torch.float32, device=device)
    F = torch.as_tensor(faces, dtype=torch.long, device=device)
    L3 = torch.as_tensor(labels[faces], device=device)
    ant = None
    if antagonist is not None:
        ant = (torch.as_tensor(antagonist[0], dtype=torch.float32, device=device), torch.as_tensor(antagonist[1], dtype=torch.long, device=device))
    m = tooth_maps_batch(V, F, L3, fdis, origins, [f // 10 in (2, 3) for f in fdis], ant, seed=seed)
    keep = ((m["tooth"] > 0).flatten(1).sum(1) >= 2000).cpu().numpy()  # 原点不在牙冠里面（牙列很乱）或者牙太小
    if keep.sum() < 4:
        return None
    sel = lambda x: x.cpu().numpy()[keep]  # noqa: E731
    return {"fdi": np.array(fdis, np.int16)[keep], "origin": np.array(origins, np.float32)[keep], "has_ant": sel(m["has_ant"]),
            **{k: sel(m[k]).astype(np.float16) for k in ("tooth", "ctx", "ant")}}


def _load_teeth3ds_raw(item):
    try:
        mesh, labels = item.load()
        return item.name, "", np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces), np.asarray(labels), None, None
    except Exception as exc:  # noqa: BLE001
        return item.name, f"{type(exc).__name__}: {exc}", None, None, None, None, None


def _load_bite_raw(args):
    """一位患者的上下颌各出一份（本颌 + 对颌，都转到本颌的标准坐标）。返回两行。"""
    zip_path, pid, labels_dir, seg_model = args
    rows = []
    try:
        from pathlib import Path

        from ..eval_crown import _labels, _load
        from .radial import _canonical

        meshes = {jaw: _load(zip_path, pid, jaw) for jaw in ("upper", "lower")}
        labels = {jaw: _labels(meshes[jaw], pid, jaw, seg_model, Path(labels_dir)) for jaw in meshes}
        for jaw, other in (("upper", "lower"), ("lower", "upper")):
            canon, rot = _canonical(meshes[jaw], labels[jaw], jaw)
            rows.append((f"{pid}_{jaw}", "", np.asarray(canon.vertices, np.float32), np.asarray(canon.faces), np.asarray(labels[jaw]),
                         (np.asarray(meshes[other].vertices) @ rot.T).astype(np.float32), np.asarray(meshes[other].faces)))
    except Exception as exc:  # noqa: BLE001
        rows.append((pid, f"{type(exc).__name__}: {exc}", None, None, None, None, None))
    return rows


def cache_gpu(jobs, loader, out_dir, workers: int = 16, device: str = "cuda") -> dict:
    """径向深度图缓存（radial.py cache / cache-bite 的 GPU 版）。loader 返回一行或多行 (名字, 错误, 顶点, 面, 标签, 对颌顶点, 对颌面)。"""
    import multiprocessing as mp
    import os
    import time
    from pathlib import Path

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, done, teeth, failed = time.time(), 0, 0, []
    with mp.get_context("spawn").Pool(workers) as pool:
        for res in pool.imap_unordered(loader, jobs):
            for name, err, v, f, lab, av, af in (res if isinstance(res, list) else [res]):
                if v is None:
                    failed.append(f"{name}: {err}")
                    continue
                data = scan_maps_gpu(v, f, lab, (av, af) if av is not None else None, seed=done, device=device)
                if data is None:
                    continue
                np.savez_compressed(out_dir / f"{name}.npz", **data)
                done += 1
                teeth += len(data["fdi"])
                if done % 100 == 0:
                    print(f"[{done}] {teeth} 颗，{time.time() - t0:.0f}s", flush=True)
    return {"jaws": done, "teeth": teeth, "seconds": round(time.time() - t0, 1), "n_failed": len(failed), "failed": failed[:5]}


def _save_mesh_cache(args):
    """CPU 进程里做：把一个颌（Bite2Text 一位患者的上下颌）解析好存成 npz，下次直接读，不用再解析 STL、自动标注、估牙弓方向。"""
    loader, job, out = args
    from pathlib import Path

    rows = loader(job)
    n = 0
    for name, err, v, f, lab, av, af in (rows if isinstance(rows, list) else [rows]):
        if v is None:
            continue
        extra = {} if av is None else {"av": av.astype(np.float32), "af": af.astype(np.int32)}
        np.savez(Path(out) / f"{name}.npz", v=v.astype(np.float32), f=f.astype(np.int32), labels=lab.astype(np.int16), **extra)
        n += 1
    return n


def _load_mesh_cache(path):
    """读解析好的网格缓存（_save_mesh_cache 写的）。"""
    from pathlib import Path

    path = Path(path)
    try:
        z = np.load(path)
        av, af = (z["av"], z["af"].astype(np.int64)) if "av" in z else (None, None)
        return path.stem, "", z["v"], z["f"].astype(np.int64), z["labels"].astype(np.int64), av, af
    except Exception as exc:  # noqa: BLE001
        return path.stem, f"{type(exc).__name__}: {exc}", None, None, None, None, None


def _with_cache_info(args):
    """D 图用的 loader：网格缓存 + 径向缓存里的牙号和原点。"""
    mesh_path, radial_path = args
    name, err, v, f, lab, av, af = _load_mesh_cache(mesh_path)
    if v is None:
        return name, err, None, None, None, None, None, None, None
    z = np.load(radial_path)
    jaw = name.rsplit("_", 1)[1]
    return name, jaw, v, f, lab, av, af, z["fdi"], z["origin"]


def dmaps_gpu(jobs, loader, out_dir, gen_model: str, workers: int = 16, device: str = "cuda") -> dict:
    """D 的输出图（和 radial_dmaps.py 同样的文件），GPU 版：CPU 进程只读文件，D 批量推理、采样、投影都在 GPU 上。"""
    import multiprocessing as mp
    import os
    import time
    from pathlib import Path

    import trimesh

    from .base import GenerationContext
    from .learned import LearnedGenerator

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = LearnedGenerator(gen_model, fallback=None, device=device)
    t0, done, teeth, failed = time.time(), 0, 0, []
    with mp.get_context("spawn").Pool(workers) as pool:
        for name, jaw, v, f, lab, av, af, fdis, origins in pool.imap_unordered(loader, jobs):
            if v is None:
                failed.append(f"{name}: {jaw}")
                continue
            mesh = trimesh.Trimesh(v, f, process=False)
            ant = trimesh.Trimesh(av, af, process=False) if av is not None else None
            res = gen.generate_many(GenerationContext(mesh=mesh, labels=lab, jaw=jaw, antagonist=ant), fdis)
            flips = [int(x) // 10 in (2, 3) for x in fdis]
            crowns = [res[int(x)].crown if int(x) in res else None for x in fdis]
            d = crowns_to_maps(crowns, origins, flips, device)
            ok = np.array([c is not None and (m > 0).sum() >= 1500 for c, m in zip(crowns, d)])
            d[~ok] = 0
            np.savez_compressed(out_dir / f"{name}.npz", fdi=fdis, d=d.astype(np.float16), has_d=ok)
            done += 1
            teeth += int(ok.sum())
            if done % 100 == 0:
                print(f"[{done}/{len(jobs)}] {teeth} 颗，{time.time() - t0:.0f}s", flush=True)
    return {"jaws": done, "teeth": teeth, "seconds": round(time.time() - t0, 1), "failed": failed[:5], "n_failed": len(failed)}


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    p = argparse.ArgumentParser(description="径向深度图数据准备的 GPU 版")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dmaps", help="D 的输出图（radial_dmaps.py 的 GPU 版）")
    d.add_argument("--radial", type=Path, required=True)
    d.add_argument("--gen-model", required=True)
    d.add_argument("--zips", type=Path)
    d.add_argument("--bite-zip")
    d.add_argument("--labels", default="../data/bite_labels")
    d.add_argument("--seg-model", default="../models/segmentation/deep.pt")
    d.add_argument("--out", type=Path, help="默认 data/radial_d/<split>")
    d.add_argument("--mesh-cache", type=Path, help="解析好的网格缓存目录（meshes 子命令生成）；给了就不再读原始文件")
    d.add_argument("--shards", type=int, default=1, help="同时开几个 GPU 进程分片做（每个约 1 GB 显存）；主进程里还有串行的 CPU 步骤，分片能成倍提速")
    d.add_argument("--workers", type=int, default=16)
    d.add_argument("--limit", type=int)
    c = sub.add_parser("cache", help="径向深度图缓存（radial.py cache / cache-bite 的 GPU 版）")
    c.add_argument("--zips", type=Path, help="Teeth3DS zip 目录")
    c.add_argument("--split", type=Path, nargs="*")
    c.add_argument("--bite-zip")
    c.add_argument("--labels", default="../data/bite_labels")
    c.add_argument("--seg-model", default="../models/segmentation/deep.pt")
    c.add_argument("--only", type=Path, help="只做这个目录里已有的那些颌（对照 CPU 版用）")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--workers", type=int, default=16)
    c.add_argument("--limit", type=int)
    m = sub.add_parser("meshes", help="把原始口扫解析好存成网格缓存（一次性，之后数据准备直接读它）")
    m.add_argument("--zips", type=Path)
    m.add_argument("--split", type=Path, nargs="*")
    m.add_argument("--bite-zip")
    m.add_argument("--labels", default="../data/bite_labels")
    m.add_argument("--seg-model", default="../models/segmentation/deep.pt")
    m.add_argument("--only", type=Path)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--workers", type=int, default=40)
    m.add_argument("--limit", type=int)
    args = p.parse_args()
    if args.cmd == "meshes":
        import multiprocessing as mp
        import os
        import time

        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        args.out.mkdir(parents=True, exist_ok=True)
        only = {f.stem for f in args.only.glob("*.npz")} if args.only else None
        if args.zips:
            from ..datasets import teeth3ds_zip_scans

            items = [s for sp in (args.split or [None]) for s in teeth3ds_zip_scans(args.zips, sp)]
            jobs = [(_load_teeth3ds_raw, s, str(args.out)) for s in items if only is None or s.name in only][: args.limit]
        else:
            import zipfile

            names = set(zipfile.ZipFile(args.bite_zip).namelist())
            pids = sorted({n.split("/")[0] for n in names if n.endswith("ios_upper.stl") and n.replace("upper", "lower") in names})
            if only is not None:
                pids = [q for q in pids if f"{q}_upper" in only or f"{q}_lower" in only]
            jobs = [(_load_bite_raw, (args.bite_zip, q, args.labels, args.seg_model), str(args.out)) for q in pids[: args.limit]]
        t0 = time.time()
        with mp.get_context("spawn").Pool(args.workers) as pool:
            n = sum(pool.imap_unordered(_save_mesh_cache, jobs))
        print(json.dumps({"saved": n, "seconds": round(time.time() - t0, 1)}), flush=True)
        return
    if args.cmd == "cache":
        only = {f.stem for f in args.only.glob("*.npz")} if args.only else None
        if args.zips:
            from ..datasets import teeth3ds_zip_scans

            items = [s for sp in (args.split or [None]) for s in teeth3ds_zip_scans(args.zips, sp)]
            items = [s for s in items if only is None or s.name in only][: args.limit]
            jobs, loader = items, _load_teeth3ds_raw
        else:
            import zipfile

            names = set(zipfile.ZipFile(args.bite_zip).namelist())
            pids = sorted({n.split("/")[0] for n in names if n.endswith("ios_upper.stl") and n.replace("upper", "lower") in names})
            if only is not None:
                pids = [q for q in pids if f"{q}_upper" in only or f"{q}_lower" in only]
            jobs, loader = [(args.bite_zip, q, args.labels, args.seg_model) for q in pids[: args.limit]], _load_bite_raw
        print(json.dumps(cache_gpu(jobs, loader, args.out, args.workers), ensure_ascii=False), flush=True)
        return
    from .radial_diffusion import d_dir_for

    caches = sorted(args.radial.glob("*.npz"))[: args.limit]
    out = args.out or d_dir_for(args.radial)
    if args.mesh_cache:
        jobs = [(str(args.mesh_cache / c.name), str(c)) for c in caches if (args.mesh_cache / c.name).exists()]
        loader = _with_cache_info
    elif args.zips:
        from ..datasets import teeth3ds_zip_scans

        items = {s.name: s for s in teeth3ds_zip_scans(args.zips)}
        jobs, loader = [(items[c.stem], str(c)) for c in caches if c.stem in items], _load_teeth3ds
    else:
        jobs, loader = [(str(c), args.bite_zip, args.labels, args.seg_model) for c in caches], _load_bite
    model = str(Path(args.gen_model).resolve())
    if args.shards <= 1:
        print(json.dumps(dmaps_gpu(jobs, loader, out, model, args.workers), ensure_ascii=False), flush=True)
        return
    import multiprocessing as mp
    import time

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.shards) as pool:
        parts = pool.starmap(_dmaps_shard, [(jobs[i:: args.shards], loader, str(out), model, max(1, args.workers // args.shards))
                                            for i in range(args.shards)])
    print(json.dumps({"jaws": sum(p["jaws"] for p in parts), "teeth": sum(p["teeth"] for p in parts), "shards": args.shards,
                      "seconds": round(time.time() - t0, 1), "n_failed": sum(p["n_failed"] for p in parts)}, ensure_ascii=False), flush=True)


def _dmaps_shard(jobs, loader, out, model, workers):
    """分片里的一个 GPU 进程（不能再开守护进程池，读文件就在本进程里做）。"""
    import os
    import time
    from pathlib import Path

    import trimesh

    from .base import GenerationContext
    from .learned import LearnedGenerator

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    gen = LearnedGenerator(model, fallback=None, device="cuda")
    t0, done, teeth, failed = time.time(), 0, 0, 0
    Path(out).mkdir(parents=True, exist_ok=True)
    for job in jobs:
        name, jaw, v, f, lab, av, af, fdis, origins = loader(job)
        if v is None:
            failed += 1
            continue
        mesh = trimesh.Trimesh(v, f, process=False)
        ant = trimesh.Trimesh(av, af, process=False) if av is not None else None
        res = gen.generate_many(GenerationContext(mesh=mesh, labels=lab, jaw=jaw, antagonist=ant), fdis)
        flips = [int(x) // 10 in (2, 3) for x in fdis]
        crowns = [res[int(x)].crown if int(x) in res else None for x in fdis]
        d = crowns_to_maps(crowns, origins, flips, "cuda")
        ok = np.array([c is not None and (m > 0).sum() >= 1500 for c, m in zip(crowns, d)])
        d[~ok] = 0
        np.savez_compressed(Path(out) / f"{name}.npz", fdi=fdis, d=d.astype(np.float16), has_d=ok)
        done += 1
        teeth += int(ok.sum())
    return {"jaws": done, "teeth": teeth, "n_failed": failed, "seconds": round(time.time() - t0, 1)}


def crowns_to_maps(crowns, origins, flips, device, n: int = 150000, seed: int = 0):
    """多个牙冠网格（trimesh，和 origins 同一坐标系）→ 径向深度图 (T, S, S)，再逐张做 clean_depth（和推理时一致）。"""
    import torch

    from .radial_diffusion import clean_depth

    g = torch.Generator(device=device).manual_seed(seed)
    pts, bs = [], []
    for k, (crown, origin, flip) in enumerate(zip(crowns, origins, flips)):
        if crown is None:
            continue
        v = torch.as_tensor(np.asarray(crown.vertices), dtype=torch.float32, device=device)
        f = torch.as_tensor(np.asarray(crown.faces), dtype=torch.long, device=device)
        s = torch.tensor([-1.0 if flip else 1.0, 1.0, 1.0], device=device)
        p = (sample_surface(v, f, n, g) - torch.as_tensor(np.asarray(origin), dtype=torch.float32, device=device)) * s
        pts.append(p)
        bs.append(torch.full((len(p),), k, device=device, dtype=torch.long))
    T = len(crowns)
    if not pts:
        return np.zeros((T, SIZE, SIZE), np.float32)
    maps = splat(torch.cat(pts), torch.cat(bs), T).cpu().numpy()
    return np.stack([clean_depth(m) if crowns[k] is not None else m for k, m in enumerate(maps)])


if __name__ == "__main__":
    main()


