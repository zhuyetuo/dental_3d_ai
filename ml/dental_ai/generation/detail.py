"""真牙细节库：给学习式生成的牙冠换上真牙的牙尖和窝沟。

学习模型（learned.py）输出的牙冠位置、大小、大体外形是准的，但表面偏平滑（回归到平均形态）。
这里从一个真牙冠库里找外形最接近的几颗，贴合到预测的牙冠上（各轴缩放 → 刚性配准 → 低频形变），
取贴合后最接近的一颗：低频形状跟着预测走，高频细节来自真牙。

库的来源：
- Teeth3DS 训练集里各牙位的牙冠（每类取一部分）；
- FDI 16 数据集（3Shape / DTU，7762 颗上颌右侧第一磨牙，get_data_fdi16.sh）：只用于 16 / 26 这一类。

    python -m dental_ai.generation.detail build --zips ../data/drive/raw/Teeth3DS --split <training_lower.txt> <training_upper.txt> \\
        --fdi16 ../data/fdi16/train --out ../data/gen_library
    GEN_LIBRARY=../data/gen_library  # 推理时 LearnedGenerator 读这个环境变量；没有就输出原来的平滑牙冠

库不随仓库分发（是数据集的衍生数据，许可不允许再分发），换机器后用上面的命令重建，几分钟。
坐标：和 learned.py 的网络输出一样（标准坐标、左侧牙已镜像成右侧），每颗牙冠以顶点质心为原点，单位 mm。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .learned import target_class

MIN_VERTS = 800  # 太小的牙冠（标注碎片、低分辨率）不进库
N_DIRS = 128


def _directions(n: int = N_DIRS) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi, theta = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]


DIRS = _directions()


def support(v: np.ndarray) -> np.ndarray:
    """支撑函数：每个方向上最远的顶点有多远。描述大小和大体外形，和顶点数、拓扑无关。"""
    return (v @ DIRS.T).max(axis=0).astype(np.float32)


def crown_submesh(vertices: np.ndarray, faces: np.ndarray, labels: np.ndarray, fdi: int):
    """标签为 fdi 的最大连通块 → (顶点, 三角形)；太小返回 None。"""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    f = faces[(labels[faces] == fdi).all(axis=1)]
    if len(f) < MIN_VERTS:
        return None
    used = np.unique(f)
    remap = np.full(len(vertices), -1, np.int64)
    remap[used] = np.arange(len(used))
    f = remap[f]
    e = np.r_[f[:, [0, 1]], f[:, [1, 2]]]
    _, comp = connected_components(sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(used), len(used))), directed=False)
    big = np.argmax(np.bincount(comp))
    keep = comp == big
    if keep.sum() < MIN_VERTS:
        return None
    remap2 = np.full(len(used), -1, np.int64)
    remap2[keep] = np.arange(keep.sum())
    f = f[keep[f].all(axis=1)]
    return vertices[used][keep].astype(np.float32), remap2[f].astype(np.int32)


def n_boundaries(f: np.ndarray) -> int:
    """开放边界的条数。完整的牙冠只有一条（牙颈线）；更多说明表面有破洞。"""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    e = np.sort(np.r_[f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=1)
    e, count = np.unique(e, axis=0, return_counts=True)
    b = e[count == 1]
    if not len(b):
        return 0
    ids, inv = np.unique(b, return_inverse=True)
    inv = inv.reshape(-1, 2)
    return int(connected_components(sp.coo_matrix((np.ones(len(inv)), (inv[:, 0], inv[:, 1])), shape=(len(ids), len(ids))), directed=False)[0])


def _unify(v: np.ndarray, f: np.ndarray, fdi: int):
    """左侧牙镜像成右侧（x 取反，三角形绕序翻转），以质心为原点。"""
    v = v.astype(np.float32).copy()
    if fdi // 10 in (2, 3):
        v[:, 0] *= -1
        f = f[:, ::-1]
    return v - v.mean(axis=0), np.ascontiguousarray(f, np.int32)


def _scan_crowns(scan):
    try:
        mesh, labels = scan.load()
        v, f, labels = np.asarray(mesh.vertices), np.asarray(mesh.faces), np.asarray(labels)
        out = []
        for fdi in sorted(int(x) for x in np.unique(labels) if x > 0 and x % 10 <= 7):
            sub = crown_submesh(v, f, labels, fdi)
            if sub is not None and n_boundaries(sub[1]) == 1:
                out.append((target_class(fdi), *_unify(*sub, fdi)))
        return out
    except Exception as exc:
        print(f"跳过 {getattr(scan, 'name', scan)}: {exc}", flush=True)
        return []


def _load_ply(path):
    import trimesh

    m = trimesh.load(path, process=False)
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32)


def _rotations(improper: bool = False):
    """24 个轴对齐的旋转（improper=True 时是 24 个带镜像的）。"""
    import itertools

    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            m = np.zeros((3, 3))
            m[np.arange(3), perm] = signs
            if (np.linalg.det(m) > 0) != improper:
                out.append(m)
    return out


def align_fdi16(crowns, reference_desc: np.ndarray) -> tuple[np.ndarray, dict]:
    """FDI 16 数据集自己的坐标系 → 库的坐标系。数据集内部已经互相对齐，所以只要一个全局旋转：
    在 24 个轴对齐的旋转里选平均支撑函数和 Teeth3DS 的 16 号牙最接近的。"""
    ref = reference_desc.mean(axis=0)
    cent = [v - v.mean(axis=0) for v, _ in crowns]

    def cost(m):
        return float(np.linalg.norm(np.mean([support(v @ m.T) for v in cent], axis=0) - ref) / np.sqrt(N_DIRS))

    scored = sorted((cost(m), i) for i, m in enumerate(_rotations()))
    mirrored = min(cost(m) for m in _rotations(improper=True))
    best = _rotations()[scored[0][1]]
    return best, {"cost_mm": round(scored[0][0], 3), "second_mm": round(scored[1][0], 3), "best_mirrored_mm": round(mirrored, 3)}


def _save_class(out: Path, cls: int, crowns, source) -> None:
    voff = np.cumsum([0] + [len(v) for v, _ in crowns])
    foff = np.cumsum([0] + [len(f) for _, f in crowns])
    np.savez(out / f"class_{cls:02d}.npz", verts=np.concatenate([v for v, _ in crowns]), faces=np.concatenate([f for _, f in crowns]),
             voff=voff, foff=foff, desc=np.stack([support(v) for v, _ in crowns]), source=np.asarray(source, np.uint8))


def build(out: Path, zips: Path | None, splits, fdi16: Path | None, per_class: int, n_fdi16: int, workers: int, seed: int = 0) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    from ..datasets import teeth3ds_zip_scans

    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    by_class: dict[int, list] = {}
    scans = [s for sp_ in (splits or [None]) for s in teeth3ds_zip_scans(zips, sp_)] if zips else []
    scans = [scans[i] for i in rng.permutation(len(scans))]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for crowns in pool.map(_scan_crowns, scans[: int(per_class * 4)], chunksize=2):  # 每个扫描只有一个颌，还要去掉有破洞的，所以多取一些
            for cls, v, f in crowns:
                if len(by_class.setdefault(cls, [])) < per_class:
                    by_class[cls].append((v, f))
    info: dict = {"per_class": {int(c): len(v) for c, v in sorted(by_class.items())}}
    source = {c: [0] * len(v) for c, v in by_class.items()}
    if fdi16 and n_fdi16 > 0:
        cls = target_class(16)
        files = sorted(Path(fdi16).glob("*_mesh.ply"))
        files = [files[i] for i in rng.permutation(len(files))[:n_fdi16]]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            raw = [c for c in pool.map(_load_ply, files, chunksize=8) if len(c[0]) >= MIN_VERTS and n_boundaries(c[1]) == 1]
        if cls not in by_class or len(by_class[cls]) < 20:
            raise ValueError("对齐 FDI 16 需要库里先有 Teeth3DS 的 16 / 26 号牙（给 --zips）")
        rot, info["fdi16_alignment"] = align_fdi16(raw[:200], np.stack([support(v) for v, _ in by_class[cls]]))
        info["fdi16_rotation"] = rot.tolist()
        for v, f in raw:
            v = (v - v.mean(axis=0)) @ rot.T
            by_class[cls].append((v.astype(np.float32), f))
            source[cls].append(1)
        info["fdi16"] = len(raw)
    for cls, crowns in by_class.items():
        _save_class(out, cls, crowns, source[cls])
    (out / "library.json").write_text(json.dumps(info, indent=2))
    return info


# ---------------------------------------------------------------- 贴合


def _kabsch(a: np.ndarray, b: np.ndarray):
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, cb - ca @ r.T


def _smooth(d: np.ndarray, faces: np.ndarray, iters: int) -> np.ndarray:
    import scipy.sparse as sp

    n = len(d)
    e = np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    a = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    a = a + a.T + sp.identity(n)
    a = sp.diags(1.0 / np.asarray(a.sum(axis=1)).ravel()) @ a
    for _ in range(iters):
        d = a @ d
    return d


def fit(v: np.ndarray, f: np.ndarray, target: np.ndarray, tree: cKDTree, smooth_iters: int = 60, max_angle_deg: float = 35.0):
    """把真牙冠 v（以质心为原点）贴合到目标点集 target（同样以质心为原点）：缩放 → 刚性 ICP → 低频形变。"""
    v = v.astype(float)
    scale = np.clip((target.max(axis=0) - target.min(axis=0)) / (v.max(axis=0) - v.min(axis=0)), 0.8, 1.25)
    x = v * scale
    x0 = x.copy()
    for _ in range(10):
        r, t = _kabsch(x, target[tree.query(x)[1]])
        x = x @ r.T + t
    r_total, _ = _kabsch(x0, x)
    if np.degrees(np.arccos(np.clip((np.trace(r_total) - 1) / 2, -1, 1))) > max_angle_deg:  # 配准跑偏了：只平移
        x = x0 + (target.mean(axis=0) - x0.mean(axis=0))
    return x + _smooth(target[tree.query(x)[1]] - x, f, smooth_iters)


def chamfer(a: np.ndarray, b: np.ndarray, ta: cKDTree | None = None, tb: cKDTree | None = None) -> float:
    return float(0.5 * ((tb or cKDTree(b)).query(a)[0].mean() + (ta or cKDTree(a)).query(b)[0].mean()))


class DetailLibrary:
    def __init__(self, root, candidates: int = 8, max_fit_mm: float = 0.25, max_miss_mm: float = 0.5):
        self.root = Path(root)
        self.candidates = candidates
        self.max_fit_mm, self.max_miss_mm = max_fit_mm, max_miss_mm
        self._classes: dict[int, dict | None] = {}

    def _class(self, cls: int):
        if cls not in self._classes:
            p = self.root / f"class_{cls:02d}.npz"
            self._classes[cls] = dict(np.load(p)) if p.exists() else None
        return self._classes[cls]

    def crown(self, cls: int, i: int):
        d = self._class(cls)
        return d["verts"][d["voff"][i]:d["voff"][i + 1]], d["faces"][d["foff"][i]:d["foff"][i + 1]]

    def refine(self, pred: np.ndarray, cls: int, sources=None):
        """pred：学习模型输出的顶点（标准坐标、已统一左右）。返回 (顶点, 三角形, 信息)；这一类没有库时返回 None。
        sources：只用这些来源（0 = Teeth3DS，1 = FDI 16），做对比实验用。"""
        d = self._class(cls)
        if d is None:
            return None
        center = pred.mean(axis=0)
        target = pred - center
        tree = cKDTree(target)
        dist = np.linalg.norm(d["desc"] - support(target), axis=1)
        if sources is not None:
            dist = np.where(np.isin(d["source"], sources), dist, np.inf)
        best = None
        for i in np.argsort(dist)[: self.candidates]:
            if not np.isfinite(dist[i]):
                break
            v, f = self.crown(cls, int(i))
            x = fit(v, f, target, tree)
            tx = cKDTree(x)
            miss = float(np.percentile(tx.query(target)[0], 95))  # 预测的牙冠上有多大一块没被真牙冠盖住（真牙冠残缺时很大）
            c = chamfer(x, target, ta=tx, tb=tree)
            if best is None or c < best[0]:
                best = (c, x, f, int(i), miss)
        if best is None or best[0] > self.max_fit_mm or best[4] > self.max_miss_mm:
            return None  # 没有贴得上的真牙冠：保留网络的平滑牙冠
        c, x, f, i, _ = best
        return x + center, f, {"detail_source": ["teeth3ds", "fdi16"][int(d["source"][i])], "detail_index": i, "detail_fit_mm": round(c, 3)}


def main() -> None:
    p = argparse.ArgumentParser(description="真牙细节库")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--zips", type=Path)
    b.add_argument("--split", type=Path, nargs="*")
    b.add_argument("--fdi16", type=Path, help="FDI 16 数据集的 train 目录（*_mesh.ply）")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--per-class", type=int, default=150, help="Teeth3DS 每个牙位取多少颗")
    b.add_argument("--n-fdi16", type=int, default=600, help="FDI 16 取多少颗")
    b.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    print(json.dumps(build(args.out, args.zips, args.split, args.fdi16, args.per_class, args.n_fdi16, args.workers), indent=2))


if __name__ == "__main__":
    main()
