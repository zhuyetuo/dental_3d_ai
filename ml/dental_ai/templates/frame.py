"""牙齿局部坐标系：x = 近中方向，y = 颊侧（唇侧）方向，z = 咬合方向，原点在牙颈线高度。

左右同名牙互为镜像，在这个坐标系下坐标相同，所以一个模板（如“下颌第一磨牙”）可以同时用于 36 和 46。
由于镜像关系，一侧的坐标系是右手系、另一侧是左手系：变换到左手系时要翻转三角形绕序（见 to_world）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from ..fdi import Jaw, arch_order
from ..geometry import ArchFrame


@dataclass
class ToothFrame:
    origin: np.ndarray
    mesial: np.ndarray
    buccal: np.ndarray
    occlusal: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        return np.stack([self.mesial, self.buccal, self.occlusal])

    @property
    def right_handed(self) -> bool:
        return float(np.linalg.det(self.matrix)) > 0

    def to_local(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        out = trimesh.Trimesh((mesh.vertices - self.origin) @ self.matrix.T, mesh.faces.copy(), process=False)
        if not self.right_handed:
            out.invert()  # 统一存成右手系下的绕序
        return out

    def to_world(self, local: trimesh.Trimesh) -> trimesh.Trimesh:
        out = trimesh.Trimesh(self.origin + local.vertices @ self.matrix, local.faces.copy(), process=False)
        if not self.right_handed:
            out.invert()
        return out


def mesial_neighbors(labels: np.ndarray, jaw: Jaw, fdi: int) -> tuple[int | None, int | None]:
    """沿牙弓离 fdi 最近的、现存的近中侧和远中侧邻牙。"""
    order = arch_order(jaw, include_third_molars=True)
    present = {int(v) for v in np.unique(labels) if v != 0}
    i = order.index(fdi)
    mid = len(order) / 2 - 0.5
    left = next((order[k] for k in range(i - 1, -1, -1) if order[k] in present), None)
    right = next((order[k] for k in range(i + 1, len(order)) if order[k] in present), None)
    # 患者右侧（order 前半）中线在 order 增大的方向
    return (right, left) if i < mid else (left, right)


def outward(arch: ArchFrame, point: np.ndarray) -> np.ndarray:
    """从牙弓中心指向该点的水平方向（颊 / 唇侧）。"""
    d = point - arch.origin
    d -= (d @ arch.occlusal) * arch.occlusal
    return d / max(np.linalg.norm(d), 1e-9)


def build_frame(arch: ArchFrame, origin: np.ndarray, mesial_direction: np.ndarray) -> ToothFrame:
    up = arch.occlusal
    m = mesial_direction - (mesial_direction @ up) * up
    m /= max(np.linalg.norm(m), 1e-9)
    b = np.cross(up, m)
    if b @ outward(arch, origin) < 0:
        b = -b
    return ToothFrame(origin=np.asarray(origin, float), mesial=m, buccal=b, occlusal=up)


def tooth_frame(mesh: trimesh.Trimesh, labels: np.ndarray, jaw: Jaw, fdi: int, arch: ArchFrame) -> ToothFrame:
    """已有牙齿的局部坐标系（用于从病例中提取模板）。原点 = 牙颈线（牙齿区域边界）的中心。"""
    from ..margin.cervical import cervical_line

    pts = mesh.vertices[labels == fdi]
    centroid = pts.mean(axis=0)
    try:
        origin = cervical_line(mesh, labels, fdi, smooth_iters=0).mean(axis=0)
    except ValueError:
        origin = centroid - arch.occlusal * (np.ptp(pts @ arch.occlusal) / 2)
    mes, dis = mesial_neighbors(labels, jaw, fdi)
    if mes is not None and dis is not None:
        # 两侧都有邻牙：中心差分近似牙弓切线。只朝近中邻牙质心看，在前牙区牙弓弯曲处偏得多
        # （用 3DTeethLand 的近远中接触点标注核对：尖牙偏 23°，改成中心差分后见 docs/real-data-test.md）
        direction = mesh.vertices[labels == mes].mean(axis=0) - mesh.vertices[labels == dis].mean(axis=0)
    elif mes is not None:
        direction = mesh.vertices[labels == mes].mean(axis=0) - centroid
    elif dis is not None:
        direction = centroid - mesh.vertices[labels == dis].mean(axis=0)
    else:
        raise ValueError(f"{fdi} 两侧都没有邻牙，无法确定近中方向")
    return build_frame(arch, origin, direction)
