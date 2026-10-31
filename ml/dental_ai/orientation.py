"""朝向归一化：不依赖分割结果，把任意朝向的单颌口扫旋转到统一坐标系。

目标坐标系与 Teeth3DS 原始数据（也就是 TGNet 训练时的坐标系）一致（右手系，单位 mm）：
    +Z  牙冠方向（上下颌都是牙冠朝 +Z：上颌模型是“翻过来、从咬合面看”的）
    +Y  后方（从切牙指向磨牙）
    +X  = Y × Z。下颌是患者左侧，上颌是患者右侧（上颌翻过来后左右对调）
    原点  网格质心

（以前写的是“上颌牙冠朝 -Z、+X 都是患者左侧”，在 Teeth3DS 的 952 个上颌扫描上核对，实际是上面这样：
两者相差绕前后轴转 180°。按旧约定转过去再交给 TGNet，上颌的分割会出错。）

步骤：
1. 咬合方向：顶点分布的最小主成分（单颌口扫是一块马蹄形的“薄片”），
   符号由面法向按面积加权投票决定（开放的口扫表面整体朝向牙冠一侧）；
2. 左右方向：在咬合平面内搜索使网格左右镜像对称程度最高的方向；
3. 前后方向：前牙区（切牙）位于中线附近，后牙区的最远点分布在两侧，据此确定哪一端是前方。

已知限制：
- 只能识别镜像对称面，左右是由右手系推出来的；如果原始扫描本身被镜像过，左右会反；
- 带底座的封闭模型（如 Poseidon3D）：法向投票失效，改用“底座底面是大片朝外的平面”判断牙冠方向，
  前后方向只用牙冠一侧的部分判断。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .fdi import Jaw
from .segmentation.base import Segmenter


@dataclass
class OrientationResult:
    transform: np.ndarray  # 4x4，把原始坐标变换到标准坐标系
    symmetry_error: float  # 镜像后的平均最近点距离（mm），越小说明左右对称面找得越准

    def apply(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        out = mesh.copy()
        out.apply_transform(self.transform)
        return out

    @property
    def inverse(self) -> np.ndarray:
        return np.linalg.inv(self.transform)


def _crown_direction(mesh: trimesh.Trimesh, axis: np.ndarray) -> tuple[np.ndarray, bool]:
    """返回 (指向牙冠一侧的咬合方向, 是否是封闭模型)。"""
    vote = float((mesh.area_faces * (mesh.face_normals @ axis)).sum())
    if abs(vote) > 0.1 * mesh.area:
        return (axis if vote > 0 else -axis), False
    # 封闭模型（带底座的模型，如 Poseidon3D）：法向投票为 0。底座底面是一大片朝外的平面
    # （实测占表面积的 16–24%，牙冠一侧只有 2–4%），牙冠在它的另一侧
    n = mesh.face_normals @ axis
    flat_pos = float(mesh.area_faces[n > 0.95].sum())
    flat_neg = float(mesh.area_faces[n < -0.95].sum())
    if max(flat_pos, flat_neg) > 2.0 * min(flat_pos, flat_neg) and max(flat_pos, flat_neg) > 0.05 * mesh.area:
        return (-axis if flat_pos > flat_neg else axis), True
    # 再退一步：开放边界（扫描边缘，位于牙龈侧）
    edges = mesh.edges_sorted
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = np.unique(uniq[counts == 1])
    if len(boundary):
        side = (mesh.vertices.mean(axis=0) - mesh.vertices[boundary].mean(axis=0)) @ axis
        return (axis if side > 0 else -axis), True
    return axis, True


def _symmetry_axis(points: np.ndarray, up: np.ndarray, steps: int = 180) -> tuple[np.ndarray, float]:
    """在垂直于 up 的平面内找左右方向：关于过质心、法向为该方向的竖直平面镜像后，与原点云最接近。

    points 用 3D 坐标而不是投影到咬合平面的坐标：牙齿的高度起伏是区分对称面的主要信息。"""
    u = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(up, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(up, u)
    tree = cKDTree(points)

    def error(d: np.ndarray) -> float:
        reflected = points - 2.0 * (points @ d)[:, None] * d
        return float(tree.query(reflected)[0].mean())

    def direction(theta: float) -> np.ndarray:
        return np.cos(theta) * u + np.sin(theta) * v

    # 粗搜索（镜像面方向只需要 0~180°），再在最优角附近细搜索
    thetas = np.linspace(0, np.pi, steps, endpoint=False)
    errs = [error(direction(t)) for t in thetas]
    best = thetas[int(np.argmin(errs))]
    step = np.pi / steps
    fine = np.linspace(best - step, best + step, 21)
    fine_errs = [error(direction(t)) for t in fine]
    t = fine[int(np.argmin(fine_errs))]
    return direction(t), float(min(fine_errs))


def normalize_orientation(
    mesh: trimesh.Trimesh,
    jaw: Jaw,
    sample: int = 4000,
    seed: int = 0,
) -> OrientationResult:
    vertices = np.asarray(mesh.vertices, dtype=float)
    center = vertices.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vertices), size=min(sample, len(vertices)), replace=False)
    pts = vertices[idx] - center

    # 1. 咬合方向
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    up, closed = _crown_direction(mesh, vt[2])  # 上下颌都是牙冠朝 +Z

    # 2. 左右方向（符号待定）
    lateral, sym_err = _symmetry_axis(pts, up)
    flat = pts - (pts @ up)[:, None] * up

    # 3. 前后方向：两端各取最远的 3% 点，离中线更近的一端是前牙。
    # 封闭模型只看牙冠一侧一半高度以上的部分：底座的轮廓（前端常是宽的弧形）会让判断反过来
    ap = np.cross(up, lateral)
    if closed:
        h = pts @ up
        flat = flat[h > h.min() + 0.5 * np.ptp(h)]
    proj = flat @ ap
    side = np.abs(flat @ lateral)
    k = max(10, len(proj) // 33)
    order = np.argsort(proj)
    low_end = side[order[:k]].mean()
    high_end = side[order[-k:]].mean()
    posterior = ap if low_end < high_end else -ap  # 低端是前牙 → +ap 指向后方

    x_axis = np.cross(posterior, up)
    rotation = np.stack([x_axis, posterior, up])  # 行向量 = 新坐标轴在原坐标中的方向
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ center
    return OrientationResult(transform=transform, symmetry_error=sym_err)


class OrientedSegmenter(Segmenter):
    """先做朝向归一化，再交给内部分割器。标签是逐顶点的，与坐标无关，不需要逆变换。"""

    def __init__(self, inner: Segmenter):
        self.inner = inner
        self.name = inner.name
        self.postprocess = inner.postprocess
        self.last: OrientationResult | None = None

    def predict(self, mesh: trimesh.Trimesh, jaw: Jaw) -> np.ndarray:
        self.last = normalize_orientation(mesh, jaw)
        return self.inner.predict(self.last.apply(mesh), jaw)
