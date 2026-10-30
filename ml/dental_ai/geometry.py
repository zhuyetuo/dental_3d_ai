"""牙弓坐标系与子网格工具。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from .fdi import Jaw, QUADRANTS, mirror, quadrant


def face_labels(mesh: trimesh.Trimesh, labels: np.ndarray) -> np.ndarray:
    """面标签 = 三个顶点标签的众数；三者都不同时取第一个顶点。"""
    fl = labels[mesh.faces]
    a, b, c = fl[:, 0], fl[:, 1], fl[:, 2]
    return np.where(b == c, b, a)


def submesh(mesh: trimesh.Trimesh, face_mask: np.ndarray) -> trimesh.Trimesh:
    faces = np.nonzero(face_mask)[0]
    if len(faces) == 0:
        return trimesh.Trimesh()
    return mesh.submesh([faces], append=True, repair=False)


def tooth_mesh(mesh: trimesh.Trimesh, labels: np.ndarray, fdi: int) -> trimesh.Trimesh:
    return submesh(mesh, face_labels(mesh, labels) == fdi)


def gum_mesh(mesh: trimesh.Trimesh, labels: np.ndarray) -> trimesh.Trimesh:
    return submesh(mesh, face_labels(mesh, labels) == 0)


def tooth_centroids(mesh: trimesh.Trimesh, labels: np.ndarray) -> dict[int, np.ndarray]:
    return {
        int(fdi): mesh.vertices[labels == fdi].mean(axis=0)
        for fdi in np.unique(labels)
        if fdi != 0
    }


@dataclass
class ArchFrame:
    """牙弓局部坐标系。

    occlusal: 咬合平面法向，指向牙冠（牙尖）方向
    lateral:  从患者右侧指向左侧
    origin:   正中矢状面上的一点
    """

    origin: np.ndarray
    occlusal: np.ndarray
    lateral: np.ndarray

    def reflect(self, points: np.ndarray) -> np.ndarray:
        """关于正中矢状面（过 origin、法向 lateral）镜像。"""
        d = (points - self.origin) @ self.lateral
        return points - 2.0 * d[:, None] * self.lateral

    def height(self, points: np.ndarray) -> np.ndarray:
        return (points - self.origin) @ self.occlusal


def estimate_arch_frame(mesh: trimesh.Trimesh, labels: np.ndarray, jaw: Jaw) -> ArchFrame:
    centroids = tooth_centroids(mesh, labels)
    if len(centroids) < 3:
        raise ValueError("牙齿数量太少，无法估计牙弓坐标系")
    pts = np.array(list(centroids.values()))

    # 牙齿质心大致共面：最小主成分即咬合平面法向
    center = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - center)
    normal = vt[2]
    # 法向朝向牙冠：牙齿顶点整体应在牙龈顶点的 +normal 侧
    tooth_h = mesh.vertices[labels != 0] @ normal
    gum_h = mesh.vertices[labels == 0] @ normal
    if len(gum_h) and tooth_h.mean() < gum_h.mean():
        normal = -normal

    # 左右方向：左侧象限质心均值 - 右侧象限质心均值
    right_q, left_q = QUADRANTS[jaw]
    right = [c for f, c in centroids.items() if quadrant(f) == right_q]
    left = [c for f, c in centroids.items() if quadrant(f) == left_q]
    if not right or not left:
        raise ValueError("需要左右两侧都有牙齿才能估计正中矢状面")
    lateral = np.mean(left, axis=0) - np.mean(right, axis=0)
    lateral -= (lateral @ normal) * normal
    lateral /= np.linalg.norm(lateral)

    # 正中矢状面位置：用左右成对存在的同名牙的中点，没有成对牙时退化为两侧均值的中点
    pairs = [(c, centroids[mirror(f)]) for f, c in centroids.items() if quadrant(f) == right_q and mirror(f) in centroids]
    if pairs:
        origin = np.mean([(a + b) / 2 for a, b in pairs], axis=0)
    else:
        origin = (np.mean(left, axis=0) + np.mean(right, axis=0)) / 2
    return ArchFrame(origin=origin, occlusal=normal, lateral=lateral)
