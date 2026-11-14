"""牙冠的“径向深度图”表示（给扩散模型用）。

从缺隙中心往四面八方看：每个方向上牙冠表面有多远，记成一张 SIZE × SIZE 的图（鱼眼投影：图中心是正上方，
越往外越朝下，最外圈是朝下 MAX_POLAR）。同样的投影下，其余牙齿（上下文）和对颌各是一张图，和牙冠的图逐像素对齐：
同一个像素 = 同一个方向，所以“这个方向上牙冠不能超过对颌 / 邻牙”对网络来说就是同一个像素上的两个数。

二维图上可以直接用图像扩散模型，分辨率比点云高得多（半径 5 mm 处一个像素约 0.14 mm），牙尖窝沟不会被平均掉。
思路参考 CrownFusion 的几何图像、DentalGEN 的多视角深度图、柴笑寒等（2026）的“挖空再补全”。

    python -m dental_ai.generation.radial roundtrip --cache ../data/gen_cache/fair_test ...   # 见 main
    python -m dental_ai.generation.radial cache --zips ../data/drive/raw/Teeth3DS --split <...> --out ../data/radial/train
    python -m dental_ai.generation.radial cache-bite --zip <Bite2Text.zip> --labels ../data/bite_labels --out ../data/radial/bite

坐标：和 learned.py 一样的标准坐标（牙冠朝 +z），原点 = 缺隙两侧最近邻牙质心的中点，左侧的牙把 x 取反。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SIZE = 160
MAX_POLAR = np.deg2rad(125.0)
R_CLIP = 16.0  # mm：上下文 / 对颌只看这么远；没有东西的像素记 0
CENTER_DROP = 3.0  # mm：投影中心比“邻牙质心的中点”低多少。0 时还原误差平均 0.17 mm（13% 的表面被挡住），2–3.5 时 0.03–0.04 mm


def pixel_directions() -> tuple[np.ndarray, np.ndarray]:
    """每个像素对应的单位方向 (SIZE, SIZE, 3)，以及像素是否在投影圆盘内。"""
    c = (np.arange(SIZE) + 0.5) / SIZE * 2 - 1
    x, y = np.meshgrid(c, c, indexing="ij")
    rho = np.hypot(x, y)
    theta, phi = rho * MAX_POLAR, np.arctan2(y, x)
    d = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
    return d.astype(np.float32), rho < 1.0


def splat(points: np.ndarray, center: np.ndarray, fill: int = 2) -> np.ndarray:
    """表面采样点 → 径向深度图：每个像素取这个方向上最近的点的距离（第一次碰到的表面）；没有点的像素是 0。
    点之间的小空洞用邻居补上（只补被有效像素包围的洞，不往外扩）。"""
    from scipy import ndimage

    v = np.asarray(points, float) - center
    r = np.linalg.norm(v, axis=1)
    keep = (r > 1e-6) & (r < R_CLIP)
    v, r = v[keep], r[keep]
    out = np.full((SIZE, SIZE), np.inf, np.float32)
    if len(r):
        theta = np.arccos(np.clip(v[:, 2] / r, -1, 1))
        phi = np.arctan2(v[:, 1], v[:, 0])
        rho = theta / MAX_POLAR
        inside = rho < 1.0
        i = np.floor((rho * np.cos(phi) * 0.5 + 0.5) * SIZE).astype(int)[inside].clip(0, SIZE - 1)
        j = np.floor((rho * np.sin(phi) * 0.5 + 0.5) * SIZE).astype(int)[inside].clip(0, SIZE - 1)
        np.minimum.at(out, (i, j), r[inside].astype(np.float32))
    valid = np.isfinite(out)
    if fill and valid.any():
        holes = ndimage.binary_closing(valid, iterations=fill) & ~valid
        for _ in range(fill):
            near = ndimage.minimum_filter(out, size=3)
            out = np.where(holes & ~np.isfinite(out), near, out)
    out[~np.isfinite(out)] = 0.0
    return out


def map_to_mesh(depth: np.ndarray, center: np.ndarray, min_r: float = 0.3):
    """径向深度图 → 开放的牙冠网格（相邻的有效像素连成三角形）。只留最大的连通块。"""
    import trimesh

    dirs, disk = pixel_directions()
    valid = disk & (depth > min_r)
    index = -np.ones((SIZE, SIZE), int)
    index[valid] = np.arange(valid.sum())
    verts = center + dirs[valid] * depth[valid][:, None]
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]
    quad = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    faces = np.concatenate([np.stack([a[quad], b[quad], c[quad]], axis=1), np.stack([a[quad], c[quad], d[quad]], axis=1)])
    mesh = trimesh.Trimesh(verts, faces, process=False)
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        mesh = max(parts, key=lambda m: len(m.faces))
    mesh.remove_unreferenced_vertices()
    if ((mesh.triangles_center - center) * mesh.face_normals).sum() < 0:
        mesh.invert()
    return mesh


def tooth_maps(mesh, labels: np.ndarray, target: int, origin: np.ndarray, antagonist=None, flip: bool = False, rng=None) -> dict:
    """标准坐标下的一个颌（mesh、labels）里，以 origin 为中心，目标牙、其余牙、对颌各一张径向深度图。"""
    import trimesh

    rng = rng or np.random.default_rng(0)
    sign = np.array([-1.0, 1.0, 1.0]) if flip else np.ones(3)
    faces = np.asarray(mesh.faces)
    verts = np.asarray(mesh.vertices)
    lab = np.asarray(labels)[faces]

    def sample(sel_faces: np.ndarray, n: int) -> np.ndarray:
        if len(sel_faces) == 0:
            return np.zeros((0, 3))
        near = np.linalg.norm(verts[sel_faces[:, 0]] - origin, axis=1) < R_CLIP + 2
        sel_faces = sel_faces[near]
        if len(sel_faces) == 0:
            return np.zeros((0, 3))
        sub = trimesh.Trimesh(verts, sel_faces, process=False)
        return (np.asarray(trimesh.sample.sample_surface(sub, n, seed=int(rng.integers(1 << 30)))[0]) - origin) * sign

    zero = np.zeros(3)
    out = {
        "tooth": splat(sample(faces[(lab == target).all(axis=1)], 150000), zero),
        "ctx": splat(sample(faces[((lab > 0) & (lab != target)).all(axis=1)], 300000), zero),
    }
    if antagonist is not None and len(antagonist.faces):
        av, af = np.asarray(antagonist.vertices), np.asarray(antagonist.faces)
        near = np.linalg.norm(av[af[:, 0]] - origin, axis=1) < R_CLIP + 2
        if near.sum() > 50:
            sub = trimesh.Trimesh(av, af[near], process=False)
            pts = (np.asarray(trimesh.sample.sample_surface(sub, 300000, seed=int(rng.integers(1 << 30)))[0]) - origin) * sign
            pts = pts[pts[:, 2] > -1.0]  # 对颌在上方；从缺隙里往下看到的不要（别的颌的牙龈）
            out["ant"] = splat(pts, zero)
    return out


def gap_origin(centroids: dict[int, np.ndarray], target: int) -> np.ndarray | None:
    """和 learned.build_context 一样：目标牙两侧最近的现存牙质心的中点。"""
    from .learned import arch_index

    ti = arch_index(target, target)
    idx = {f: arch_index(f, target) for f in centroids if f != target and (f // 10 in (1, 2)) == (target // 10 in (1, 2))}
    anchors = []
    after = [f for f, i in idx.items() if i > ti]
    before = [f for f, i in idx.items() if i < ti]
    if after:
        anchors.append(centroids[min(after, key=lambda f: idx[f])])
    if before:
        anchors.append(centroids[max(before, key=lambda f: idx[f])])
    if not anchors:
        return None
    # 往牙龈方向降一点：从牙冠靠下的位置往外看，咬合面和轴面都在视野里，被挡住的部分最少（roundtrip 上试出来的）
    return np.mean(anchors, axis=0) - np.array([0.0, 0.0, CENTER_DROP])


def scan_maps(mesh, labels: np.ndarray, antagonist=None, seed: int = 0) -> dict | None:
    """一个标准坐标下的颌：每颗牙（不含第三磨牙）挖掉后的三张图。返回要存盘的数组。"""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    verts = np.asarray(mesh.vertices)
    centroids = {int(f): verts[labels == f].mean(axis=0) for f in np.unique(labels) if f > 0 and (labels == f).sum() >= 200}
    rows = {"fdi": [], "origin": [], "tooth": [], "ctx": [], "ant": [], "has_ant": []}
    for fdi in sorted(centroids):
        if fdi % 10 > 7:
            continue
        origin = gap_origin(centroids, fdi)
        if origin is None:
            continue
        maps = tooth_maps(mesh, labels, fdi, origin, antagonist, flip=fdi // 10 in (2, 3), rng=rng)
        if (maps["tooth"] > 0).sum() < 2000:  # 原点不在牙冠里面（牙列很乱）或者牙太小
            continue
        rows["fdi"].append(fdi)
        rows["origin"].append(origin)
        rows["tooth"].append(maps["tooth"])
        rows["ctx"].append(maps["ctx"])
        rows["ant"].append(maps.get("ant", np.zeros((SIZE, SIZE), np.float32)))
        rows["has_ant"].append("ant" in maps)
    if len(rows["fdi"]) < 4:
        return None
    return {"fdi": np.array(rows["fdi"], np.int16), "origin": np.array(rows["origin"], np.float32), "has_ant": np.array(rows["has_ant"]),
            **{k: np.stack(rows[k]).astype(np.float16) for k in ("tooth", "ctx", "ant")}}


def _canonical(mesh, labels, jaw):
    import trimesh

    from ..geometry import estimate_arch_frame
    from .learned import canonical_rotation

    rot = canonical_rotation(estimate_arch_frame(mesh, labels, jaw), jaw)
    return trimesh.Trimesh(np.asarray(mesh.vertices) @ rot.T, mesh.faces, process=False), rot


def _cache_zip_one(args):
    item, out, seed = args
    try:
        dest = Path(out) / f"{item.name}.npz"
        if dest.exists():
            return 1
        mesh, labels = item.load()
        data = scan_maps(mesh, labels, seed=seed)  # Teeth3DS 本来就是标准坐标
        if data is None:
            return 0
        np.savez_compressed(dest, **data)
        return 1
    except Exception as exc:
        print(f"跳过 {getattr(item, 'name', item)}: {type(exc).__name__}: {exc}", flush=True)
        return 0


def _cache_bite_one(args):
    zip_path, pid, seg_model, labels_dir, out = args
    try:
        from ..eval_crown import _labels, _load

        out = Path(out)
        if all((out / f"{pid}_{jaw}.npz").exists() for jaw in ("upper", "lower")):
            return 2
        meshes = {jaw: _load(zip_path, pid, jaw) for jaw in ("upper", "lower")}
        labels = {jaw: _labels(meshes[jaw], pid, jaw, seg_model, Path(labels_dir)) for jaw in meshes}
        n = 0
        for i, (jaw, other) in enumerate((("upper", "lower"), ("lower", "upper"))):
            import trimesh

            canon, rot = _canonical(meshes[jaw], labels[jaw], jaw)
            ant = trimesh.Trimesh(np.asarray(meshes[other].vertices) @ rot.T, meshes[other].faces, process=False)
            data = scan_maps(canon, labels[jaw], ant, seed=i)
            if data is not None:
                np.savez_compressed(out / f"{pid}_{jaw}.npz", **data)
                n += 1
        return n
    except Exception as exc:
        print(f"跳过 {pid}: {type(exc).__name__}: {exc}", flush=True)
        return 0


def _run_pool(fn, jobs, workers):
    import multiprocessing as mp
    import os

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    done = 0
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, n in enumerate(pool.imap_unordered(fn, jobs), 1):
            done += n
            if i % 50 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {done}", flush=True)
    return done


def roundtrip(data_dir: Path, limit: int) -> dict:
    """表示本身丢了多少：真牙 → 径向深度图 → 网格，和真牙比（mm）。"""
    import trimesh

    from ..geometry import tooth_mesh
    from ..io import load_labels, load_mesh
    from ..meshdist import closest_point

    err, hidden = [], []
    for obj in sorted(data_dir.rglob("*.obj"))[:limit]:
        gt = obj.with_suffix(".json")
        if not gt.exists():
            continue
        mesh, labels = load_mesh(obj), np.asarray(load_labels(gt)[0])
        verts = np.asarray(mesh.vertices)
        centroids = {int(f): verts[labels == f].mean(axis=0) for f in np.unique(labels) if f > 0 and (labels == f).sum() >= 200}
        for fdi in [f for f in centroids if f % 10 in (4, 6, 7)][:4]:
            origin = gap_origin(centroids, fdi)
            if origin is None:
                continue
            real = tooth_mesh(mesh, labels, fdi)
            depth = tooth_maps(mesh, labels, fdi, origin)["tooth"]
            if (depth > 0).sum() < 2000:
                continue
            back = map_to_mesh(depth, origin)
            pts = np.asarray(trimesh.sample.sample_surface(real, 8000, seed=0)[0])
            d = closest_point(back, pts)[1]
            err.append([d.mean(), np.percentile(d, 95), d.max()])
            hidden.append(float((d > 0.3).mean()))
    err = np.array(err)
    return {"teeth": len(err), "mean_mm": round(float(err[:, 0].mean()), 3), "p95_mm": round(float(np.median(err[:, 1])), 3),
            "max_mm_median": round(float(np.median(err[:, 2])), 3), "share_over_0.3mm": round(float(np.mean(hidden)), 4)}


def main() -> None:
    global CENTER_DROP, MAX_POLAR
    p = argparse.ArgumentParser(description="牙冠的径向深度图表示")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("roundtrip")
    r.add_argument("--data-dir", type=Path, required=True)
    r.add_argument("--limit", type=int, default=20)
    r.add_argument("--drop", type=float, nargs="*", default=[CENTER_DROP])
    r.add_argument("--polar", type=float, nargs="*", default=[float(np.rad2deg(MAX_POLAR))])
    c = sub.add_parser("cache")
    c.add_argument("--zips", type=Path, required=True)
    c.add_argument("--split", type=Path, nargs="*")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--workers", type=int, default=32)
    b = sub.add_parser("cache-bite")
    b.add_argument("--zip", type=Path, required=True)
    b.add_argument("--labels", type=Path, required=True)
    b.add_argument("--seg-model", default="../models/segmentation/deep.pt")
    b.add_argument("--exclude", type=Path)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--workers", type=int, default=24)
    b.add_argument("--limit", type=int)
    args = p.parse_args()
    if args.cmd == "roundtrip":
        for drop in args.drop:
            for polar in args.polar:
                CENTER_DROP, MAX_POLAR = drop, np.deg2rad(polar)
                print(json.dumps({"drop": drop, "polar": polar, **roundtrip(args.data_dir, args.limit)}, ensure_ascii=False), flush=True)
    elif args.cmd == "cache":
        from ..datasets import teeth3ds_zip_scans

        args.out.mkdir(parents=True, exist_ok=True)
        jobs = []
        for s in args.split or [None]:
            jobs += [(scan, str(args.out), i) for i, scan in enumerate(teeth3ds_zip_scans(args.zips, s))]
        print(f"{len(jobs)} 个扫描", flush=True)
        print("完成", _run_pool(_cache_zip_one, jobs, args.workers))
    else:
        import zipfile

        names = set(zipfile.ZipFile(args.zip).namelist())
        pids = sorted({n.split("/")[0] for n in names if n.endswith("ios_upper.stl") and n.replace("upper", "lower") in names})
        if args.exclude:
            skip = {f.stem.rsplit("_", 1)[0] for f in args.exclude.glob("*.npy")}
            pids = [q for q in pids if q not in skip]
        pids = pids[: args.limit]
        args.out.mkdir(parents=True, exist_ok=True)
        print(f"{len(pids)} 位患者", flush=True)
        print("完成", _run_pool(_cache_bite_one, [(str(args.zip), q, args.seg_model, str(args.labels), str(args.out)) for q in pids], args.workers))


if __name__ == "__main__":
    main()
