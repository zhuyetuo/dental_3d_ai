"""学习式缺失牙生成（模板变形网络，第一版）。

训练数据自动构造：从带标注的完整牙弓里“挖掉”一颗牙，网络看其余牙齿（点云 + 每个点的牙位），把一个球面模板
变形成被挖掉的那颗牙。输出是固定拓扑的封闭网格（球面细分），坐标单位 mm。

    python -m dental_ai.generation.learned cache --zips ../data/drive/raw/Teeth3DS --split <training_lower.txt> <training_upper.txt> --out ../data/gen_cache/train
    python -m dental_ai.generation.learned cache --data-dir ../data/fair_test --out ../data/gen_cache/fair_test
    python -m dental_ai.generation.learned train --cache ../data/gen_cache/train --out ../models/generation/toothgen.pt
    python -m dental_ai.generation.learned eval --model ../models/generation/toothgen.pt --cache ../data/gen_cache/fair_test --out ../runs/toothgen/eval_fair_test

坐标系：原点 = 缺隙两侧最近邻牙质心的中点（只有一侧有邻牙时就是那颗牙的质心），推理时也能算出来；
目标牙在左侧时把 x 取反，统一成“目标牙在右侧”，网络只需要学 2 个颌 × 7 个牙位 = 14 类（不生成第三磨牙）。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

TOOTH_POINTS = 2048  # 每颗牙缓存的点数
RIM_POINTS = 128  # 每颗牙缓存的边界（牙颈线）点数
CTX_PER_TOOTH = 1024
N_CTX = 2048  # 输入的上下文点数
MIN_VERTS = 200  # 顶点太少的牙不要（标注碎片）
SCALE = 10.0  # 网络内部坐标 = mm / SCALE
N_ANT = 2048  # 输入的对颌点数
ANT_RADIUS = 10.0  # 对颌只取原点周围（咬合面内）这么远的部分，mm
ANT_KEEP = 100000  # 每个颌缓存的对颌点数上限
ANT_SLAB = 4.0  # 对颌只取最靠近本颌的这么厚一层，mm
NB_POINTS = 512  # 每颗紧邻的牙取的点数（只在训练的损失里用）


def shape_terms(pred, gt, nb, has_nb):
    """整体形状的两类量，训练时当损失、验证时当指标（都是 mm，每个样本一个数）：
    extent：生成牙和真牙在 x、y、z 三个方向上的最大 / 最小坐标差多少（外形尺寸、位置、高度）；
    gap：生成牙到近中 / 远中邻牙的最小距离，和真牙到邻牙的最小距离差多少（没有那一侧邻牙的不算）。"""
    import torch

    extent = (torch.cat([pred.max(dim=1)[0], pred.min(dim=1)[0]], dim=1) - torch.cat([gt.max(dim=1)[0], gt.min(dim=1)[0]], dim=1)).abs().mean(dim=1)
    gap = torch.zeros_like(extent)
    for k in range(2):
        d_pred = torch.cdist(pred, nb[:, k]).amin(dim=(1, 2))
        d_real = torch.cdist(gt, nb[:, k]).amin(dim=(1, 2))
        gap = gap + (d_pred - d_real).abs() * has_nb[:, k]
    return extent, gap / has_nb.sum(dim=1).clamp(min=1)


def antagonist_points(vertices: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """标准坐标下的对颌顶点和法向 → (K, 6)。法向统一成朝向本颌（标准坐标里本颌牙冠朝 +z，对颌的咬合面朝 −z）：
    有的口扫三角形绕序是反的。"""
    normals = np.asarray(normals, np.float32)
    if len(normals) and normals[:, 2].mean() > 0:
        normals = -normals
    return np.concatenate([np.asarray(vertices, np.float32), normals], axis=1)


def crop_antagonist(ant: np.ndarray | None, origin: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    """原点周围 ANT_RADIUS（只看 x、y）以内的对颌点，随机取 N_ANT 个并减去原点；太少（对颌缺牙、没扫到）返回 None。"""
    if ant is None or not len(ant):
        return None
    near = ant[np.linalg.norm(ant[:, :2] - origin[:2], axis=1) < ANT_RADIUS]
    if len(near) < 64:
        return None
    # 只留朝着本颌的那一层（最低处往上 ANT_SLAB mm）：对颌的牙龈、舌侧更高的部分不要。训练缓存里只有离牙 6 mm 以内的对颌点，
    # 流程里给的是整块对颌网格，不这样统一的话模型看到的点分布不一样（评分卡上量到过：牙冠平均高出 0.6 mm）
    near = near[near[:, 2] < np.percentile(near[:, 2], 1) + ANT_SLAB]
    out = near[rng.choice(len(near), N_ANT, replace=len(near) < N_ANT)].copy()
    out[:, :3] -= origin
    return out


# ---------------------------------------------------------------- 缓存


def _tooth_points(mesh, labels, seed: int):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    referenced = np.zeros(len(mesh.vertices), bool)
    referenced[np.unique(mesh.faces)] = True
    # 标签边界上的顶点：所在三角形里有别的标签（牙龈或邻牙）。每颗牙的这些点就是它的边界环（牙颈线，邻面处升到接触点）
    faces = np.asarray(mesh.faces)
    fl = labels[faces]
    mixed = faces[(fl != fl[:, :1]).any(axis=1)]
    on_border = np.zeros(len(labels), bool)
    on_border[mixed.reshape(-1)] = True
    fdis, pts, rims = [], [], []
    for fdi in sorted(int(f) for f in np.unique(labels) if f > 0):
        idx = np.flatnonzero((labels == fdi) & referenced)
        border = np.flatnonzero((labels == fdi) & on_border)
        if len(idx) < MIN_VERTS or len(border) < 8:
            continue
        pick = rng.choice(idx, TOOTH_POINTS, replace=len(idx) < TOOTH_POINTS)
        fdis.append(fdi)
        pts.append(np.asarray(mesh.vertices)[pick].astype(np.float32))
        rims.append(np.asarray(mesh.vertices)[rng.choice(border, RIM_POINTS, replace=len(border) < RIM_POINTS)].astype(np.float32))
    if not pts:
        return np.zeros(0, np.int16), np.zeros((0, TOOTH_POINTS, 3), np.float32), np.zeros((0, RIM_POINTS, 3), np.float32)
    return np.array(fdis, np.int16), np.stack(pts), np.stack(rims)


def _cache_one(args):
    kind, item, out_dir, seed = args
    try:
        if kind == "zip":
            name = item.name
            dest = Path(out_dir) / f"{name}.npz"
            if dest.exists():
                return name
            mesh, labels = item.load()
        else:
            from ..io import load_labels, load_mesh

            obj, gt = item
            name = Path(obj).stem
            dest = Path(out_dir) / f"{name}.npz"
            if dest.exists():
                return name
            mesh, labels = load_mesh(obj), load_labels(gt)[0]
        fdi, pts, rim = _tooth_points(mesh, labels, seed)
        if len(fdi) < 4:
            return None
        np.savez_compressed(dest, fdi=fdi, pts=pts, rim=rim)
        return name
    except Exception as exc:
        print(f"跳过 {item}: {exc}", flush=True)
        return None


def build_cache(out: Path, zips: Path | None, splits: list[Path] | None, data_dir: Path | None, workers: int) -> int:
    from concurrent.futures import ProcessPoolExecutor

    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if zips:
        from ..datasets import teeth3ds_zip_scans

        for s in splits or [None]:
            jobs += [("zip", scan, out, i) for i, scan in enumerate(teeth3ds_zip_scans(zips, s))]
    if data_dir:
        for i, obj in enumerate(sorted(data_dir.rglob("*.obj"))):
            gt = obj.with_suffix(".json")
            if gt.exists():
                jobs.append(("dir", (str(obj), str(gt)), out, i))
    print(f"预处理 {len(jobs)} 个扫描", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        done = [n for n in pool.map(_cache_one, jobs, chunksize=2) if n]
    return len(done)


def _cache_bite_one(args):
    zip_path, pid, seg_model, out, labels_dir = args
    try:
        import trimesh
        from scipy.spatial import cKDTree

        from ..eval_crown import _labels, _load
        from ..geometry import estimate_arch_frame

        out = Path(out)
        if all((out / f"{pid}_{jaw}.npz").exists() for jaw in ("upper", "lower")):
            return 2
        meshes = {jaw: _load(zip_path, pid, jaw) for jaw in ("upper", "lower")}
        labels = {jaw: _labels(meshes[jaw], pid, jaw, seg_model, Path(labels_dir)) for jaw in meshes}
        rng = np.random.default_rng(0)
        n = 0
        for i, (jaw, other) in enumerate((("upper", "lower"), ("lower", "upper"))):
            mesh, lab = meshes[jaw], labels[jaw]
            # 和推理时一样：先转到标准坐标（牙冠 +z），对颌跟着转
            rot = canonical_rotation(estimate_arch_frame(mesh, lab, jaw), jaw)
            canon = trimesh.Trimesh(np.asarray(mesh.vertices) @ rot.T, mesh.faces, process=False)
            fdi, pts, rim = _tooth_points(canon, lab, seed=i)
            if len(fdi) < 6:
                continue
            ov = np.asarray(meshes[other].vertices) @ rot.T
            # 范围留得比 crop_antagonist 用到的大（圆盘半径 + 一层的厚度），这样训练时取到的点和流程里从整块对颌网格取到的一样
            idx = np.flatnonzero(np.isfinite(cKDTree(pts.reshape(-1, 3)).query(ov, distance_upper_bound=12.0)[0]))
            if len(idx) > ANT_KEEP:
                idx = rng.choice(idx, ANT_KEEP, replace=False)
            ant = antagonist_points(ov[idx], np.asarray(meshes[other].vertex_normals)[idx] @ rot.T)
            np.savez_compressed(out / f"{pid}_{jaw}.npz", fdi=fdi, pts=pts, rim=rim, ant=ant)
            n += 1
        return n
    except Exception as exc:
        print(f"跳过 {pid}: {type(exc).__name__}: {exc}", flush=True)
        return 0


def build_bite_cache(zip_path: Path, seg_model: str, labels_dir: Path, out: Path, workers: int, exclude: Path | None = None, limit: int | None = None) -> int:
    import multiprocessing as mp
    import os
    import zipfile

    names = set(zipfile.ZipFile(zip_path).namelist())
    pids = sorted({n.split("/")[0] for n in names if n.endswith("ios_upper.stl") and n.replace("upper", "lower") in names})
    if exclude:
        skip = {f.stem.rsplit("_", 1)[0] for f in Path(exclude).glob("*.npy")}
        pids = [q for q in pids if q not in skip]
    pids = pids[:limit]
    out.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(pids)} 位患者", flush=True)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "2")
    done = 0
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, n in enumerate(pool.imap_unordered(_cache_bite_one, [(str(zip_path), q, seg_model, str(out), str(labels_dir)) for q in pids]), 1):
            done += n
            if i % 20 == 0 or i == len(pids):
                print(f"[{i}/{len(pids)}] {done} 个颌", flush=True)
    return done


# ---------------------------------------------------------------- 样本


def arch_index(fdi: int, target: int) -> int:
    """牙弓上的位置：目标牙所在一侧 8..1 → 0..7，对侧 1..8 → 8..15。"""
    q, p = divmod(int(fdi), 10)
    return 8 - p if q == target // 10 else 7 + p


def target_class(fdi: int) -> int:
    q, p = divmod(int(fdi), 10)
    return (0 if q in (1, 2) else 8) + p - 1


def build_context(fdi: np.ndarray, pts: np.ndarray, target: int, rng: np.random.Generator):
    """其余牙齿（fdi (T,)、pts (T, TOOTH_POINTS, 3)）→ 上下文点 (N_CTX,3)（已减去原点）、每个点的牙弓位置、原点。
    原点 = 目标牙两侧最近的现存牙质心的中点（只有一侧时就是那一颗）。训练和推理共用。"""
    idx = np.array([arch_index(f, target) for f in fdi])
    ti = arch_index(target, target)
    cent = pts.mean(axis=1)
    anchors = []
    if (idx > ti).any():
        anchors.append(cent[np.argmin(np.where(idx > ti, idx, 99))])
    if (idx < ti).any():
        anchors.append(cent[np.argmax(np.where(idx < ti, idx, -1))])
    if not anchors:
        raise ValueError("目标牙两侧都没有现存牙")
    origin = np.mean(anchors, axis=0)
    sub = rng.integers(0, TOOTH_POINTS, (len(fdi), CTX_PER_TOOTH))
    ctx = pts[np.arange(len(fdi))[:, None], sub].reshape(-1, 3)
    ctx_idx = np.repeat(idx, CTX_PER_TOOTH)
    pick = rng.choice(len(ctx), N_CTX, replace=len(ctx) < N_CTX)
    return ctx[pick] - origin, ctx_idx[pick], origin


def make_sample(fdi: np.ndarray, pts: np.ndarray, rim: np.ndarray, t: int, rng: np.random.Generator, drop: bool, augment: bool,
                ant: np.ndarray | None = None, ant_drop: float = 0.0):
    """挖掉第 t 颗牙。返回 ctx (N_CTX,3) mm、ctx_idx (N_CTX,)、目标类别、真牙点 (TOOTH_POINTS,3) mm、真牙边界点 (RIM_POINTS,3) mm、
    对颌点和法向 (N_ANT,6)、有没有对颌；坐标已平移到原点并统一左右。ant 是这个颌缓存的对颌点 (K,6)，没有就是 None。"""
    target = int(fdi[t])
    keep = [i for i in range(len(fdi)) if i != t]
    if drop and len(keep) > 3 and rng.random() < 0.35:  # 再随机去掉 1–2 颗（模拟多颗缺牙、对侧也缺）
        for i in rng.choice(keep, rng.integers(1, 3), replace=False):
            keep.remove(int(i))
    ctx, ctx_idx, origin = build_context(fdi[keep], pts[keep], target, rng)
    gt = np.concatenate([pts[t], rim[t]]) - origin  # 真牙点和边界点一起变换，返回时再拆开
    # 近中、远中紧邻的牙（留在上下文里的才算）：训练时用来约束“和邻牙的间隙要和真牙一样”
    idx_all = np.array([arch_index(f, target) for f in fdi])
    ti = arch_index(target, target)
    nb, has_nb = np.zeros((2, NB_POINTS, 3)), np.zeros(2, np.float32)
    for k, want in enumerate((ti - 1, ti + 1)):
        j = [i for i in keep if idx_all[i] == want]
        if j:
            nb[k] = pts[j[0]][rng.integers(0, TOOTH_POINTS, NB_POINTS)] - origin
            has_nb[k] = 1
    a6 = None if (ant_drop and rng.random() < ant_drop) else crop_antagonist(ant, origin, rng)
    has_ant = a6 is not None
    if a6 is None:
        a6 = np.zeros((N_ANT, 6), np.float32)
    if target // 10 in (2, 3):  # 左侧 → 镜像成右侧
        ctx, gt, nb = ctx * [-1, 1, 1], gt * [-1, 1, 1], nb * [-1, 1, 1]
        a6 = a6 * np.array([-1, 1, 1, -1, 1, 1], np.float32)
    if augment:
        a, b, c = np.deg2rad(rng.uniform(-10, 10)), np.deg2rad(rng.uniform(-5, 5)), np.deg2rad(rng.uniform(-5, 5))
        rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
        ry = np.array([[np.cos(c), 0, np.sin(c)], [0, 1, 0], [-np.sin(c), 0, np.cos(c)]])
        r = (rz @ rx @ ry).T
        m = r * rng.uniform(0.95, 1.05)
        ctx, gt, nb = ctx @ m, gt @ m, nb @ m
        a6 = np.concatenate([a6[:, :3] @ m, a6[:, 3:] @ r], axis=1)
    gt = gt.astype(np.float32)
    return (ctx.astype(np.float32), ctx_idx.astype(np.int64), target_class(target), gt[:TOOTH_POINTS], gt[TOOTH_POINTS:],
            a6.astype(np.float32), np.float32(has_ant), nb.astype(np.float32), has_nb)


def load_cache(cache: Path | list[Path]):
    """一个或几个缓存目录。每个扫描：(名字, fdi, pts, rim, 对颌点 (K,6) 或 None)。"""
    scans = []
    for c in [cache] if isinstance(cache, (str, Path)) else cache:
        for f in sorted(Path(c).glob("*.npz")):
            with np.load(f) as d:
                ant = d["ant"].astype(np.float32) if "ant" in d.files and len(d["ant"]) else None
                scans.append((f.stem, d["fdi"].astype(np.int64), d["pts"].astype(np.float32), d["rim"].astype(np.float32), ant))
    items = [(s, t) for s, sc in enumerate(scans) for t in range(len(sc[1])) if sc[1][t] % 10 <= 7]
    return scans, items


def _dataset(scans, items, train: bool, ant_drop: float = 0.0, use_ant: bool = True):
    import torch

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            s, t = items[i]
            rng = np.random.default_rng() if train else np.random.default_rng(i)
            ctx, ctx_idx, cls, gt, rim, ant, has_ant, nb, has_nb = make_sample(
                scans[s][1], scans[s][2], scans[s][3], t, rng, drop=train, augment=train,
                ant=scans[s][4] if use_ant else None, ant_drop=ant_drop if train else 0.0)
            return ctx, ctx_idx, cls, gt, rim, ant, has_ant, nb, has_nb, i

    return DS()


# ---------------------------------------------------------------- 模型


def dome_template(rings: int = 40, segments: int = 64, max_polar_deg: float = 120.0):
    """开口的穹顶（球冠）：极点朝 +z（牙冠顶），最后一圈是唯一的边界环（对应牙颈线），按方位角排列。

    封闭球面去拟合开口的牙冠曲面会瘪成双层薄壳（体积接近 0），所以模板和牙冠用同样的拓扑。"""
    verts = [[0.0, 0.0, 1.0]]
    for i in range(1, rings + 1):
        th = np.deg2rad(max_polar_deg) * i / rings
        for j in range(segments):
            ph = 2 * np.pi * j / segments
            verts.append([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    faces = [[0, 1 + j, 1 + (j + 1) % segments] for j in range(segments)]
    for i in range(1, rings):
        a, b = 1 + (i - 1) * segments, 1 + i * segments
        for j in range(segments):
            j2 = (j + 1) % segments
            faces += [[a + j, b + j, b + j2], [a + j, b + j2, a + j2]]
    boundary = np.arange(1 + (rings - 1) * segments, 1 + rings * segments)
    return np.array(verts, np.float32), np.array(faces, np.int64), boundary


def build_model(k: int = 16, latent: int = 1024, subdivisions: int = 4, template: str = "sphere", antagonist: bool = False, refine: bool = False):
    """antagonist：多一个看对颌的分支（缺隙上方的对颌表面点 + 法向 → 256 维，和牙弓上下文的特征拼在一起）。
    refine：粗结果出来后，每个顶点再看离它最近的几个对颌点，预测一个小的修正量（让咬合面去适应对颌的尖窝）。"""
    import torch
    import trimesh

    nn = torch.nn
    if template == "dome":
        tv, tf, boundary = dome_template()
        sphere = trimesh.Trimesh(tv, tf, process=False)
    else:
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions)
        boundary = np.zeros(0, np.int64)
    interior = np.ones(len(sphere.vertices), bool)
    interior[boundary] = False
    degree = np.bincount(sphere.edges_unique.reshape(-1), minlength=len(sphere.vertices)).astype(np.float32)

    def knn(x, k):  # (B, 3, N) → (B, N, k)
        d = torch.cdist(x.transpose(1, 2), x.transpose(1, 2))
        return d.topk(k, dim=-1, largest=False)[1]

    def edge_features(x, idx):
        b, c, n = x.shape
        flat = (idx + torch.arange(b, device=x.device).view(-1, 1, 1) * n).view(-1)
        xt = x.transpose(2, 1).contiguous()
        neigh = xt.view(b * n, -1)[flat].view(b, n, -1, c)
        center = xt.view(b, n, 1, c).expand(-1, -1, neigh.shape[2], -1)
        return torch.cat([neigh - center, center], dim=3).permute(0, 3, 1, 2).contiguous()

    class EdgeConv(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.mlp = nn.Sequential(nn.Conv2d(2 * cin, cout, 1, bias=False), nn.GroupNorm(8, cout), nn.LeakyReLU(0.2))

        def forward(self, x, idx):
            return self.mlp(edge_features(x, idx)).max(dim=-1)[0]

    class ToothGen(nn.Module):
        def __init__(self):
            super().__init__()
            self.k = k
            self.emb_idx, self.emb_cls = nn.Embedding(16, 16), nn.Embedding(16, 16)
            self.e1, self.e2, self.e3 = EdgeConv(3 + 32, 64), EdgeConv(64, 128), EdgeConv(128, 256)
            self.glob = nn.Sequential(nn.Conv1d(448, 1024, 1, bias=False), nn.GroupNorm(16, 1024), nn.LeakyReLU(0.2))
            self.antagonist, self.refine = antagonist, refine and antagonist
            self.fc = nn.Sequential(nn.Linear(2048 + (256 if antagonist else 0), latent), nn.LeakyReLU(0.2))
            if antagonist:
                self.ant_mlp = nn.Sequential(nn.Conv1d(6, 64, 1), nn.LeakyReLU(0.2), nn.Conv1d(64, 128, 1), nn.LeakyReLU(0.2),
                                             nn.Conv1d(128, 256, 1), nn.LeakyReLU(0.2))
                self.ant_fc = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(0.2))
                self.no_ant = nn.Parameter(torch.zeros(256))  # 没有对颌时用的特征
            if self.refine:
                self.ref_local = nn.Sequential(nn.Conv2d(7, 64, 1), nn.LeakyReLU(0.2), nn.Conv2d(64, 64, 1), nn.LeakyReLU(0.2))
                self.ref_z = nn.Linear(latent, 128)
                self.ref_dec = nn.Sequential(nn.Conv1d(3 + 64 + 128 + 16, 256, 1), nn.LeakyReLU(0.2),
                                             nn.Conv1d(256, 128, 1), nn.LeakyReLU(0.2), nn.Conv1d(128, 3, 1))
                nn.init.zeros_(self.ref_dec[-1].weight)  # 修正量从 0 开始
                nn.init.zeros_(self.ref_dec[-1].bias)
            self.dec = nn.Sequential(
                nn.Conv1d(3 + latent + 16, 512, 1), nn.LeakyReLU(0.2),
                nn.Conv1d(512, 512, 1), nn.LeakyReLU(0.2),
                nn.Conv1d(512, 256, 1), nn.LeakyReLU(0.2),
                nn.Conv1d(256, 3, 1),
            )
            self.register_buffer("template", torch.tensor(sphere.vertices, dtype=torch.float32))
            self.register_buffer("faces", torch.tensor(sphere.faces, dtype=torch.long))
            self.register_buffer("edges", torch.tensor(sphere.edges_unique, dtype=torch.long))
            self.register_buffer("interior", torch.tensor(interior))
            self.register_buffer("degree", torch.tensor(degree))
            self.register_buffer("boundary", torch.tensor(boundary, dtype=torch.long))

        def laplacian(self, pred):  # 每个顶点减去邻居平均（均匀拉普拉斯），只看非边界顶点
            e0, e1 = self.edges[:, 0], self.edges[:, 1]
            s = torch.zeros_like(pred).index_add(1, e0, pred[:, e1]).index_add(1, e1, pred[:, e0])
            return (pred - s / self.degree.view(1, -1, 1))[:, self.interior]

        def refine_step(self, out, ant, t, z, c):  # out (B,V,3) mm、ant (B,K,6) → 修正量 (B,V,3) mm
            v = out.shape[1]
            idx = torch.cdist(out, ant[..., :3]).topk(8, dim=2, largest=False)[1]  # (B,V,8)
            nb = torch.gather(ant.unsqueeze(1).expand(-1, v, -1, -1), 2, idx.unsqueeze(-1).expand(-1, -1, -1, 6))
            off = out.unsqueeze(2) - nb[..., :3]  # 对颌点 → 顶点，mm
            sd = (off * nb[..., 3:]).sum(-1, keepdim=True)  # 沿对颌法向的距离：负的是穿进去了
            f = torch.cat([off.clamp(-5, 5), nb[..., 3:], sd.clamp(-5, 5)], dim=-1).permute(0, 3, 1, 2)
            local = self.ref_local(f).max(dim=-1)[0]
            h = torch.cat([t, local, self.ref_z(z).unsqueeze(-1).expand(-1, -1, v), c.unsqueeze(-1).expand(-1, -1, v)], dim=1)
            return self.ref_dec(h).transpose(1, 2)

        def forward(self, ctx, ctx_idx, cls, ant=None, has_ant=None):  # ctx (B, N, 3) mm、ant (B, K, 6) → (B, V, 3) mm
            b, n, _ = ctx.shape
            xyz = (ctx / SCALE).transpose(1, 2)
            c = self.emb_cls(cls)
            x = torch.cat([xyz, self.emb_idx(ctx_idx).transpose(1, 2), c.unsqueeze(-1).expand(-1, -1, n)], dim=1)
            idx = knn(xyz, self.k)
            x1 = self.e1(x, idx)
            x2 = self.e2(x1, idx)
            x3 = self.e3(x2, idx)
            g = self.glob(torch.cat([x1, x2, x3], dim=1))
            feats = [g.max(dim=-1)[0], g.mean(dim=-1)]
            if self.antagonist:
                if ant is None:
                    feats.append(self.no_ant.unsqueeze(0).expand(b, -1))
                else:
                    if has_ant is None:
                        has_ant = torch.ones(b, device=ctx.device)
                    ha = self.ant_mlp(torch.cat([ant[..., :3] / SCALE, ant[..., 3:]], dim=-1).transpose(1, 2))
                    a = self.ant_fc(torch.cat([ha.max(dim=-1)[0], ha.mean(dim=-1)], dim=1))
                    m = has_ant.view(-1, 1)
                    feats.append(m * a + (1 - m) * self.no_ant)
            z = self.fc(torch.cat(feats, dim=1))
            v = self.template.shape[0]
            t = self.template.t().unsqueeze(0).expand(b, -1, -1)
            h = torch.cat([t, z.unsqueeze(-1).expand(-1, -1, v), c.unsqueeze(-1).expand(-1, -1, v)], dim=1)
            out = (0.3 * t + self.dec(h)).transpose(1, 2) * SCALE
            if self.refine and ant is not None:
                out = out + self.refine_step(out, ant, t, z, c) * has_ant.view(-1, 1, 1)
            return out

    return ToothGen()


def chamfer(pred, gt):
    """(B,V,3)、(B,M,3) → 每个样本的双向平均距离（mm）。"""
    import torch

    d = torch.cdist(pred, gt)
    return (d.min(dim=2)[0].mean(dim=1) + d.min(dim=1)[0].mean(dim=1)) / 2


def antagonist_distance(pred, ant):
    """(B,V,3)、对颌点和法向 (B,K,6) → 每个顶点沿最近对颌点法向的距离 (B,V)，mm；负的是穿进对颌了。离对颌 3 mm 以外的记 +inf。
    用的是点而不是曲面，只是近似（训练时的惩罚项和粗略的验证指标）；正式的数字看评分卡。"""
    import torch

    dist, idx = torch.cdist(pred, ant[..., :3]).min(dim=2)
    p = torch.gather(ant, 1, idx.unsqueeze(-1).expand(-1, -1, 6))
    sd = ((pred - p[..., :3]) * p[..., 3:]).sum(-1)
    return torch.where(dist < 3.0, sd, torch.full_like(sd, float("inf")))


# ---------------------------------------------------------------- 训练 / 评估


def load_model(path: Path, device: str):
    import torch

    ckpt = torch.load(path, map_location=device)
    model = build_model(template=ckpt.get("template", "sphere"), antagonist=ckpt.get("antagonist", False), refine=ckpt.get("refine", False)).to(device)
    missing = model.load_state_dict(ckpt["state"], strict=False).missing_keys
    # 早期保存的模型没有这几个由模板决定的缓冲区，其余权重必须齐全
    assert set(missing) <= {"interior", "degree", "boundary"}, missing
    return model.eval(), ckpt


def train(cache: Path | list[Path], out: Path, epochs: int, batch_size: int, lr: float, workers: int, max_scans: int | None, edge_weight: float,
          template: str = "sphere", init: Path | None = None, lap_weight: float = 0.0, rim_weight: float = 1.0,
          antagonist: bool = False, refine: bool = False, ant_drop: float = 0.3, pen_weight: float = 10.0, clear_weight: float = 0.0,
          extent_weight: float = 0.0, gap_weight: float = 0.0) -> dict:
    import torch

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scans, _ = load_cache(cache)
    scans = scans[:max_scans]
    order = np.random.default_rng(0).permutation(len(scans))
    n_val = max(1, int(len(scans) * 0.05))
    val_set = set(order[:n_val].tolist())
    items = [(s, t) for s, sc in enumerate(scans) for t in range(len(sc[1])) if sc[1][t] % 10 <= 7]
    tr =[it for it in items if it[0] not in val_set]
    va = [it for it in items if it[0] in val_set]
    print(json.dumps({"train_scans": len(scans) - n_val, "val_scans": n_val, "train_samples": len(tr), "val_samples": len(va), "device": device}), flush=True)
    dl = torch.utils.data.DataLoader(_dataset(scans, tr, True, ant_drop, antagonist), batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True, persistent_workers=workers > 0)
    dv = torch.utils.data.DataLoader(_dataset(scans, va, False, 0.0, antagonist), batch_size=batch_size, num_workers=min(workers, 4))
    model = build_model(template=template, antagonist=antagonist, refine=refine).to(device)
    if init:  # 沿用已训练的权重（模板不同时，形状对不上的缓冲区跳过）
        own = model.state_dict()
        src = torch.load(init, map_location=device)["state"]
        old = {k: v for k, v in src.items() if k in own and v.shape == own[k].shape}
        w = src.get("fc.0.weight")
        if w is not None and w.shape != own["fc.0.weight"].shape:  # 原模型没有对颌分支：新加的输入列从 0 开始，一开始和原模型输出一样
            grown = torch.zeros_like(own["fc.0.weight"])
            grown[:, : w.shape[1]] = w
            old["fc.0.weight"] = grown
        model.load_state_dict(old, strict=False)
        print(json.dumps({"init": str(init), "loaded_tensors": len(old), "of": len(own)}), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl), pct_start=0.05)
    out.parent.mkdir(parents=True, exist_ok=True)
    best, t0 = float("inf"), time.time()

    def score(val):
        return val["chamfer_mm"] + (rim_weight * val["rim_mm"] if len(model.boundary) else 0.0)

    if init:  # 接着训练：先量一下起点，之后只有比它好才覆盖 out（out 可以就是 init）
        val = evaluate(model, dv, device)
        best = score(val)
        print(json.dumps({"epoch": 0, "val_chamfer_mm": round(val["chamfer_mm"], 4), "val_rim_mm": round(val["rim_mm"], 4), "note": "起点"}), flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for ctx, ctx_idx, cls, gt, rim, ant, has_ant, nb, has_nb, _ in dl:
            ctx, ctx_idx, cls, gt, rim = ctx.to(device), ctx_idx.to(device), cls.to(device), gt.to(device), rim.to(device)
            ant, has_ant = ant.to(device), has_ant.to(device)
            pred = model(ctx, ctx_idx, cls, ant, has_ant) if antagonist else model(ctx, ctx_idx, cls)
            cd = chamfer(pred, gt).mean()
            if extent_weight or gap_weight:  # 整体形状：外形尺寸 / 高度、和邻牙的间隙都向真牙看齐
                extent, gap = shape_terms(pred, gt, nb.to(device), has_nb.to(device))
                cd = cd + extent_weight * extent.mean() + gap_weight * gap.mean()
            if antagonist and has_ant.sum() > 0 and (pen_weight or clear_weight):
                sd = antagonist_distance(pred, ant)
                if pen_weight:  # 穿进对颌的部分：按穿入深度罚（对所有顶点取平均）
                    cd = cd + pen_weight * (torch.relu(-sd).mean(dim=1) * has_ant).sum() / has_ant.sum()
                if clear_weight:
                    # 只罚穿入的话模型学会的是“躲开对颌”（留缝）。再要求离对颌最近的地方和真牙一样近：
                    # 生成牙到对颌的最小距离（软最小值）对齐真牙到对颌的最小距离
                    soft = -0.02 * torch.logsumexp(-torch.where(torch.isfinite(sd), sd, torch.full_like(sd, 3.0)) / 0.02, dim=1)
                    real = antagonist_distance(gt, ant).min(dim=1)[0]
                    ok = (torch.isfinite(real) & (has_ant > 0)).float()
                    real = torch.where(torch.isfinite(real), real, torch.zeros_like(real)).clamp(-0.3, 1.0)
                    cd = cd + clear_weight * ((soft - real).abs() * ok).sum() / ok.sum().clamp(min=1)
            if rim_weight and len(model.boundary):  # 模板的边界环对齐真牙的边界环，否则穹顶会折回去变成双层
                cd = cd + rim_weight * chamfer(pred[:, model.boundary], rim).mean()
            e = pred[:, model.edges[:, 0]] - pred[:, model.edges[:, 1]]
            loss = cd + edge_weight * (e**2).sum(dim=-1).mean()
            if lap_weight:
                loss = loss + lap_weight * (model.laplacian(pred) ** 2).sum(dim=-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total, n = total + cd.item(), n + 1
        val = evaluate(model, dv, device)
        row = {"epoch": epoch, "train_loss": round(total / n, 4), "val_chamfer_mm": round(val["chamfer_mm"], 4),
               "val_rim_mm": round(val["rim_mm"], 4), "val_area_mm2": round(val["area_mm2"], 1), "minutes": round((time.time() - t0) / 60, 1)}
        row.update({f"val_{k}": round(v, 4) for k, v in {**val.get("shape", {}), **val.get("occlusion", {})}.items()})
        print(json.dumps(row), flush=True)
        ckpt = {"state": model.state_dict(), "epoch": epoch, "val": val, "template": template, "antagonist": antagonist, "refine": refine}
        torch.save(ckpt, out.with_name(out.stem + ".last.pt"))  # 最后一轮总是留着（验证分数没超过起点时 out 不会被覆盖）
        if score(val) < best:
            best = score(val)
            torch.save(ckpt, out)
    info = {"best_val_score": best, "epochs": epochs, "train_samples": len(tr)}
    print(json.dumps(info), flush=True)
    return info


def evaluate(model, loader, device, collect: bool = False):
    import torch

    model.eval()
    cds, cens, rows, areas, clear, with_ant, shape = [], [], [], [], [], [], []
    f = model.faces
    use_ant = getattr(model, "antagonist", False)
    with torch.no_grad():
        for ctx, ctx_idx, cls, gt, rim, ant, has_ant, nb, has_nb, i in loader:
            ant, has_ant = ant.to(device), has_ant.to(device)
            pred = model(ctx.to(device), ctx_idx.to(device), cls.to(device), ant, has_ant) if use_ant else model(ctx.to(device), ctx_idx.to(device), cls.to(device))
            gt = gt.to(device)
            cd = chamfer(pred, gt)
            extent, gap = shape_terms(pred, gt, nb.to(device), has_nb.to(device))
            shape.append(torch.stack([extent, gap], dim=1).cpu())
            # 有对颌的样本：生成的牙到对颌的最小距离（近似），和真牙点到对颌的最小距离
            c_pred, c_real = antagonist_distance(pred, ant).min(dim=1)[0], antagonist_distance(gt, ant).min(dim=1)[0]
            clear.append(torch.stack([c_pred, c_real], dim=1).cpu()), with_ant.append(has_ant.cpu() > 0)
            # 边界环到真牙边界的距离（开口模板才有意义）；封闭模板没有边界，记 0
            cen = chamfer(pred[:, model.boundary], rim.to(device)) if len(model.boundary) else torch.zeros_like(cd)
            a = pred[:, f[:, 1]] - pred[:, f[:, 0]]
            b = pred[:, f[:, 2]] - pred[:, f[:, 0]]
            areas.append((torch.linalg.cross(a, b).norm(dim=-1).sum(dim=1) / 2).cpu())
            cds.append(cd.cpu()), cens.append(cen.cpu())
            if collect:
                rows += [(int(a), int(b), float(c), float(d)) for a, b, c, d in zip(i, cls, cd.cpu(), cen.cpu())]
    out = {"chamfer_mm": float(torch.cat(cds).mean()), "rim_mm": float(torch.cat(cens).mean()),
           "area_mm2": float(torch.cat(areas).mean()), "n": int(sum(len(c) for c in cds))}
    sh = torch.cat(shape)
    out["shape"] = {"extent_mm": float(sh[:, 0].mean()), "gap_mm": float(sh[:, 1].mean())}
    sel = torch.cat(with_ant)
    if sel.any():
        c = torch.cat(clear)[sel]
        ok = torch.isfinite(c).all(dim=1)  # 两边都在对颌 3 mm 以内的才比
        p, r = c[ok, 0], c[ok, 1]
        out["occlusion"] = {
            "ant_n": int(sel.sum()), "ant_chamfer_mm": float(torch.cat(cds)[sel].mean()),
            "clear_med_mm": float(p.median()) if len(p) else 0.0, "real_clear_med_mm": float(r.median()) if len(r) else 0.0,
            "clear_err_mm": float((p - r).abs().median()) if len(p) else 0.0,  # 生成牙和真牙到对颌的最小距离差多少
            "pen_rate": float((p < -0.1).float().mean()) if len(p) else 0.0,
            "open_rate": float((p > 0.3).float().mean()) if len(p) else 0.0,
            "real_pen_rate": float((r < -0.1).float().mean()) if len(r) else 0.0,
            "real_open_rate": float((r > 0.3).float().mean()) if len(r) else 0.0,
        }
    if collect:
        out["rows"] = rows
    return out


GROUPS = {"incisor": (1, 2), "canine": (3,), "premolar": (4, 5), "molar": (6, 7)}


def run_eval(model_path: Path, cache: Path, out: Path, batch_size: int, n_samples: int, hide_antagonist: bool = False) -> dict:
    import torch
    import trimesh

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(model_path, device)
    scans, items = load_cache(cache)
    ds = _dataset(scans, items, False)
    if hide_antagonist:  # 消融：对颌只用来量距离，不给模型看
        inner = model.forward
        model.forward = lambda ctx, ctx_idx, cls, ant=None, has_ant=None: inner(ctx, ctx_idx, cls)
    res = evaluate(model, torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=4), device, collect=True)
    rows = res.pop("rows")
    summary = {"n_scans": len(scans), "n_teeth": res["n"], "model_epoch": ckpt["epoch"], "template": ckpt.get("template", "sphere"), "all": {"chamfer_mm": res["chamfer_mm"], "rim_mm": res["rim_mm"], "area_mm2": res["area_mm2"],
               "chamfer_median_mm": float(np.median([r[2] for r in rows])), "chamfer_p90_mm": float(np.percentile([r[2] for r in rows], 90))}}
    for jaw, off in (("upper", 0), ("lower", 8)):
        for name, pos in GROUPS.items():
            sel = [r for r in rows if r[1] - off + 1 in pos]
            if sel:
                summary[f"{jaw}_{name}"] = {"n": len(sel), "chamfer_mm": float(np.mean([r[2] for r in sel])), "rim_mm": float(np.mean([r[3] for r in sel]))}
    if "occlusion" in res:
        summary["occlusion"] = res["occlusion"]
    summary["antagonist_model"] = bool(getattr(model, "antagonist", False)) and not hide_antagonist
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    model.eval()
    faces = model.faces.cpu().numpy()
    step = max(1, len(ds) // max(n_samples, 1))
    (out / "samples").mkdir(exist_ok=True)
    for j in range(0, len(ds), step)[:n_samples]:
        ctx, ctx_idx, cls, gt, _, ant, has_ant, _, _, _ = ds[j]
        with torch.no_grad():
            pred = model(torch.tensor(ctx)[None].to(device), torch.tensor(ctx_idx)[None].to(device), torch.tensor([cls]).to(device),
                         torch.tensor(ant)[None].to(device), torch.tensor([has_ant]).to(device))[0].cpu().numpy()
        if has_ant:
            trimesh.PointCloud(ant[:, :3]).export(out / "samples" / f"{scans[items[j][0]][0]}_{int(scans[items[j][0]][1][items[j][1]])}_antagonist.ply")
        s, t = items[j]
        stem = f"{scans[s][0]}_{int(scans[s][1][t])}"
        trimesh.Trimesh(pred, faces, process=False).export(out / "samples" / f"{stem}_pred.ply")
        trimesh.PointCloud(gt).export(out / "samples" / f"{stem}_real.ply")
        trimesh.PointCloud(ctx).export(out / "samples" / f"{stem}_context.ply")
    print(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------- 接入流程


def canonical_rotation(arch, jaw: str) -> np.ndarray:
    """世界坐标 → 训练用的标准坐标（Teeth3DS：牙冠 +z、磨牙 +y；下颌患者左侧 +x，上颌是翻过来的，患者右侧 +x）。
    返回 R（行向量是标准坐标轴），标准坐标 = 世界坐标 @ R.T。"""
    ez = np.asarray(arch.occlusal, float)
    ex = np.asarray(arch.lateral, float) * (1.0 if jaw == "lower" else -1.0)
    ex = ex - (ex @ ez) * ez
    ex /= np.linalg.norm(ex)
    return np.stack([ex, np.cross(ez, ex), ez])


class LearnedGenerator:
    """模板变形网络生成缺失牙，实现 ToothGenerator 接口。第三磨牙、上下文不足等不支持的情况交给 fallback。"""

    name = "learned"

    def __init__(self, model_path, fallback=None, device: str | None = None, detail_dir=None):
        import os

        self.model_path = Path(model_path)
        self.fallback = fallback
        self.device = device or os.environ.get("GEN_DEVICE")
        self._model = None
        # 真牙细节库（generation/detail.py）：有就给生成的牙冠换上真牙的牙尖窝沟，没有就输出网络的平滑牙冠
        self.detail_dir = detail_dir or os.environ.get("GEN_LIBRARY")
        src = os.environ.get("GEN_LIBRARY_SOURCES")  # 实验开关：只用部分来源，如 "0"（Teeth3DS）
        self.detail_sources = [int(s) for s in src.split(",")] if src else None
        self._detail = None

    @property
    def library(self):  # 预备体牙冠的外形供体仍用模板库
        return getattr(self.fallback, "library", None)

    def _load(self):
        if self._model is None:
            import torch

            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model, ckpt = load_model(self.model_path, self.device)
            if ckpt.get("template", "sphere") != "dome":
                raise ValueError("接入流程需要开口模板（--template dome）训练的模型")
        return self._model

    def _detail_library(self):
        if self._detail is None and self.detail_dir and Path(self.detail_dir).is_dir():
            from .detail import DetailLibrary

            self._detail = DetailLibrary(self.detail_dir)
        return self._detail

    def _fall(self, ctx, target, reason: str):
        if self.fallback is None:
            raise ValueError(f"{target.fdi} 学习模型不支持：{reason}")
        tooth = self.fallback.generate(ctx, target)
        tooth.info["learned_fallback"] = reason
        return tooth

    def generate(self, ctx, target):
        import torch
        import trimesh

        from ..geometry import estimate_arch_frame
        from .base import GeneratedTooth

        if target.kind == "closed":
            raise ValueError(f"{target.fdi} 的间隙已关闭，不生成")
        if target.fdi % 10 == 8:
            return self._fall(ctx, target, "第三磨牙")
        arch = estimate_arch_frame(ctx.mesh, ctx.labels, ctx.jaw)
        rot = canonical_rotation(arch, ctx.jaw)
        labels = np.asarray(ctx.labels).copy()
        labels[np.isin(labels, list(ctx.excluded_donors) + [target.fdi])] = 0  # 预备体形态被磨改过，不当上下文
        canon = trimesh.Trimesh(np.asarray(ctx.mesh.vertices) @ rot.T, ctx.mesh.faces, process=False)
        fdi, pts, _ = _tooth_points(canon, labels, seed=0)
        same_jaw = np.array([(f // 10 in (1, 2)) == (target.fdi // 10 in (1, 2)) for f in fdi], bool)
        fdi, pts = fdi[same_jaw], pts[same_jaw]
        if len(fdi) < 2:
            return self._fall(ctx, target, "现存牙太少")
        try:
            c, c_idx, origin = build_context(fdi, pts, target.fdi, np.random.default_rng(0))
        except ValueError as exc:
            return self._fall(ctx, target, str(exc))
        flip = target.fdi // 10 in (2, 3)
        sign = np.array([-1.0, 1.0, 1.0]) if flip else np.ones(3)
        model = self._load()
        a6 = None
        if getattr(model, "antagonist", False) and ctx.antagonist is not None and len(ctx.antagonist.vertices):
            # 带对颌分支的模型：缺隙上方的对颌表面也给它看，生成时就知道该多高、尖窝朝哪
            a6 = crop_antagonist(antagonist_points(np.asarray(ctx.antagonist.vertices) @ rot.T, np.asarray(ctx.antagonist.vertex_normals) @ rot.T),
                                 origin, np.random.default_rng(0))
        extra = () if a6 is None else (torch.tensor((a6 * np.concatenate([sign, sign])).astype(np.float32))[None].to(self.device),
                                       torch.ones(1, device=self.device))
        with torch.no_grad():
            pred = model(
                torch.tensor((c * sign).astype(np.float32))[None].to(self.device),
                torch.tensor(c_idx.astype(np.int64))[None].to(self.device),
                torch.tensor([target_class(target.fdi)]).to(self.device),
                *extra,
            )[0].cpu().numpy().astype(float)
        faces = model.faces.cpu().numpy()
        detail_info = {}
        detail = self._detail_library()
        if detail is not None and target.fdi % 10 != 7:  # 第二磨牙不换：库里的第二磨牙常常扫描不全，换上去误差明显变大
            refined = detail.refine(pred, target_class(target.fdi), self.detail_sources)
            if refined is not None:
                pred, faces, detail_info = refined
        verts = (pred * sign + origin) @ rot
        if flip:  # 镜像后三角形绕序要翻转，否则法向朝里
            faces = faces[:, ::-1]
        crown = trimesh.Trimesh(verts, faces, process=False)
        if ((crown.triangles_center - verts.mean(axis=0)) * crown.face_normals).sum() < 0:
            crown.invert()
        source = f"learned:{self.model_path.name}" + (f"+detail:{detail_info['detail_source']}" if detail_info else "")
        info = {"source": source, "placement": "learned", "gap_mm": target.gap_mm, "occlusal_drop_mm": 0.0,
                "antagonist_input": a6 is not None, **detail_info}
        # 咬合由 occlusion.adjust_occlusion 在流程里处理（不再整体下移）
        return GeneratedTooth(fdi=target.fdi, crown=crown, method=self.name, info=info)


def make_generator(model_path=None, template_dir=None, preferences=None):
    """有学习模型（参数或环境变量 GEN_MODEL）就用它、几何方法兜底；没有就用原来的对侧镜像 / 模板。"""
    import os

    from .mirror_template import MirrorTemplateGenerator

    base = MirrorTemplateGenerator(template_dir=template_dir, preferences=preferences)
    model_path = model_path or os.environ.get("GEN_MODEL")
    if model_path and Path(model_path).exists():
        learned = LearnedGenerator(model_path, fallback=base)
        # 有扩散模型（RADIAL_MODEL）时：D 出整体形状，扩散模型从它加噪到 RADIAL_REFINE 再去噪、只补细节（低频保留 D 的）。
        # 评分卡（60 位患者）上整体形状不变，相对邻牙高度、咬合接触面积、挤进邻牙都更接近天然牙，见 docs/results.md
        radial = os.environ.get("RADIAL_MODEL")
        if radial and Path(radial).exists():
            from .radial_diffusion import RadialDiffusionGenerator

            return RadialDiffusionGenerator(radial, fallback=learned, refine=float(os.environ.get("RADIAL_REFINE", "0.6")),
                                            keep_sigma=float(os.environ.get("RADIAL_KEEP_SIGMA", "6")))
        return learned
    return base


def main() -> None:
    p = argparse.ArgumentParser(description="学习式缺失牙生成（模板变形网络）")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--zips", type=Path)
    c.add_argument("--split", type=Path, nargs="*")
    c.add_argument("--data-dir", type=Path)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--workers", type=int, default=32)
    b = sub.add_parser("cache-bite", help="咬合对齐的上下颌口扫（Bite2Text）→ 带对颌的训练缓存；牙的范围用分割模型自动标")
    b.add_argument("--zip", type=Path, required=True)
    b.add_argument("--seg-model", required=True)
    b.add_argument("--labels", type=Path, required=True, help="自动标注的缓存目录")
    b.add_argument("--exclude", type=Path, help="这个目录里出现过的患者不要（评分卡用的患者，文件名 <患者>_<颌>.npy）")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--workers", type=int, default=6)
    b.add_argument("--limit", type=int)
    t = sub.add_parser("train")
    t.add_argument("--cache", type=Path, required=True, nargs="+")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--epochs", type=int, default=80)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--workers", type=int, default=12)
    t.add_argument("--max-scans", type=int)
    t.add_argument("--edge-weight", type=float, default=0.2)
    t.add_argument("--template", choices=["sphere", "dome"], default="dome")
    t.add_argument("--init", type=Path, help="从已有权重接着训练（可以和 --out 是同一个文件：只有验证集上更好才覆盖）；建议同时把 --lr 调小，如 3e-4")
    t.add_argument("--lap-weight", type=float, default=0.5, help="拉普拉斯平滑项（防止折叠）")
    t.add_argument("--rim-weight", type=float, default=1.0, help="模板边界环对齐真牙边界环（牙颈线）")
    t.add_argument("--antagonist", action="store_true", help="加对颌分支（缓存里要有带对颌的扫描，见 cache-bite）")
    t.add_argument("--refine", action="store_true", help="再加一步逐顶点的修正（看最近的对颌点）")
    t.add_argument("--ant-drop", type=float, default=0.3, help="训练时随机不给对颌的比例（没有对颌时也要能用）")
    t.add_argument("--pen-weight", type=float, default=10.0, help="穿入对颌的惩罚")
    t.add_argument("--clear-weight", type=float, default=0.0, help="到对颌的最小距离对齐真牙（0 = 不用）")
    t.add_argument("--extent-weight", type=float, default=0.0, help="外形尺寸、位置、高度（三个方向的最大 / 最小坐标）对齐真牙")
    t.add_argument("--gap-weight", type=float, default=0.0, help="到近中 / 远中邻牙的最小距离对齐真牙")
    e = sub.add_parser("eval")
    e.add_argument("--model", type=Path, required=True)
    e.add_argument("--cache", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--samples", type=int, default=16)
    e.add_argument("--hide-antagonist", action="store_true", help="不给模型看对颌（消融）")
    args = p.parse_args()
    if args.cmd == "cache":
        print(f"完成 {build_cache(args.out, args.zips, args.split, args.data_dir, args.workers)} 个扫描 → {args.out}")
    elif args.cmd == "cache-bite":
        print(f"完成 {build_bite_cache(args.zip, args.seg_model, args.labels, args.out, args.workers, args.exclude, args.limit)} 个颌 → {args.out}")
    elif args.cmd == "train":
        train(args.cache, args.out, args.epochs, args.batch_size, args.lr, args.workers, args.max_scans, args.edge_weight,
              template=args.template, init=args.init, lap_weight=args.lap_weight, rim_weight=args.rim_weight,
              antagonist=args.antagonist, refine=args.refine, ant_drop=args.ant_drop, pen_weight=args.pen_weight, clear_weight=args.clear_weight,
              extent_weight=args.extent_weight, gap_weight=args.gap_weight)
    else:
        run_eval(args.model, args.cache, args.out, args.batch_size, args.samples, args.hide_antagonist)


if __name__ == "__main__":
    main()
