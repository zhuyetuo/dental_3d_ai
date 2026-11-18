"""反复算的中间结果存硬盘（data/cache/<种类>/<钥匙>.npz）。钥匙 = 输入内容的指纹 + 模型文件的版本（路径、大小、修改时间）+ 参数，
输入或模型一变钥匙就变，不会读到过期的结果。

- CachingSegmenter：分割结果（我们的深度模型、TeethSplineSeg 服务……），真实缺牙评估、二选一反复用；
- cached_preparation：模拟预备体 + 棱强度，预备体评分卡、颈缘线调参反复用。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2] / "data/cache"


def mesh_key(mesh: trimesh.Trimesh, *extra) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(np.asarray(mesh.vertices, np.float32)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(mesh.faces, np.int64)).tobytes())
    for e in extra:
        h.update(json.dumps(e, sort_keys=True, default=str).encode() if not isinstance(e, (bytes, np.ndarray)) else
                 (e if isinstance(e, bytes) else np.ascontiguousarray(e).tobytes()))
    return h.hexdigest()


def file_version(path) -> str:
    """模型文件的版本：路径 + 大小 + 修改时间（换了权重钥匙就变）。不是文件（如服务地址）就用它本身。"""
    p = Path(str(path))
    if p.exists():
        st = p.stat()
        return f"{p.resolve()}:{st.st_size}:{int(st.st_mtime)}"
    return str(path)


def load(kind: str, key: str):
    f = ROOT / kind / f"{key}.npz"
    if f.exists():
        try:
            return dict(np.load(f, allow_pickle=False))
        except Exception:  # noqa: BLE001  坏文件当没有
            return None
    return None


def save(kind: str, key: str, **arrays) -> None:
    d = ROOT / kind
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{key}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(d / f"{key}.npz")  # 先写临时文件再改名：并行写同一个钥匙也不会坏


class CachingSegmenter:
    """包一层：同一个网格、同一个模型（版本）、同一个颌只分割一次。"""

    def __init__(self, inner, model_id: str):
        self.inner, self.model_id = inner, model_id
        self.name = getattr(inner, "name", "cached")

    def __getattr__(self, item):  # last（朝向变换）等属性透传
        return getattr(self.inner, item)

    def predict(self, mesh, jaw):
        key = mesh_key(mesh, self.model_id, jaw)
        hit = load("seg", key)
        if hit is not None:
            return hit["labels"].astype(np.int64)
        labels = np.asarray(self.inner.predict(mesh, jaw))
        save("seg", key, labels=labels.astype(np.int16))
        return labels


def cached_preparation(mesh, labels, jaw, fdi, **kw):
    """synthetic.simulate_preparation + 预备体网格的棱强度，按输入内容缓存。返回 (预备体网格, 真值颈缘线, 棱强度)。"""
    from .margin.detect import edge_sharpness
    from .synthetic import simulate_preparation

    key = mesh_key(mesh, np.asarray(labels, np.int16), jaw, fdi, kw, "prep_v1")
    hit = load("prep_sim", key)
    if hit is not None:
        return trimesh.Trimesh(hit["verts"], np.asarray(mesh.faces), process=False), hit["truth"], hit["sharp"]
    prep, truth = simulate_preparation(mesh, labels, jaw, fdi, **kw)
    sharp = edge_sharpness(prep)
    save("prep_sim", key, verts=np.asarray(prep.vertices), truth=np.asarray(truth), sharp=np.asarray(sharp, np.float32))
    return prep, truth, sharp
