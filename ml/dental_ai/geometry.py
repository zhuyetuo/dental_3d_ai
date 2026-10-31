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


def tooth_mesh(mesh: trimesh.Trimesh, labels: np.ndarray, fdi: int, clean: bool = True) -> trimesh.Trimesh:
    crown = submesh(mesh, face_labels(mesh, labels) == fdi)
    return clean_crown(crown) if clean and len(crown.faces) else crown


def clean_crown(crown: trimesh.Trimesh) -> trimesh.Trimesh:
    """把按标签切出来的牙冠整理成“一个开放边界（牙颈）”的曲面。

    真实口扫上，切出来的牙冠有几十到几百个小洞（扫描本身的小洞、个别三角形归了牙龈），
    还有脱离主体的小碎片，牙颈边界也常有夹点（边界自己碰到自己）。只保留面积最大的连通块，
    除最长的边界环（牙颈）外的环都用扇形补上。"""
    parts = crown.split(only_watertight=False)
    if len(parts) > 1:
        crown = max(parts, key=lambda m: m.area)
    crown = trimesh.Trimesh(crown.vertices.copy(), crown.faces.copy(), process=False)
    crown.remove_unreferenced_vertices()

    from .margin.cervical import boundary_loops  # 避免循环导入

    loops = boundary_loops(crown)  # 按三角形绕序、在夹点处拆开的简单环，最长的是牙颈
    if len(loops) <= 1:
        return crown
    vertices, new_faces = [crown.vertices], [crown.faces]
    n = len(crown.vertices)
    for loop in loops[1:]:
        center = n + len(vertices) - 1
        vertices.append(crown.vertices[loop].mean(axis=0)[None])
        # 环的走向与所在三角形的绕序相同，补洞的三角形要反过来
        new_faces.append(np.column_stack([np.roll(loop, -1), loop, np.full(len(loop), center)]))
    return trimesh.Trimesh(np.vstack(vertices), np.vstack(new_faces), process=False)


def fill_small_holes(mesh: trimesh.Trimesh, max_perimeter: float = 15.0) -> trimesh.Trimesh:
    """补上周长不超过 max_perimeter（mm）的洞，只加三角形、不加顶点，逐顶点标签仍然对得上。

    真实口扫（如 Teeth3DS）有几千个零点几毫米的小洞，显示时会透出背景，满屏黑点。扫描外缘很长，不受影响。"""
    from .margin.cervical import boundary_loops  # 避免循环导入

    new_faces = [mesh.faces]
    for loop in boundary_loops(mesh):
        p = mesh.vertices[loop]
        if np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1).sum() > max_perimeter:
            continue
        # 以第一个顶点为扇心；环的走向与所在三角形的绕序相同，补洞的三角形要反过来
        i = np.arange(1, len(loop) - 1)
        new_faces.append(np.column_stack([loop[i + 1], loop[i], np.full(len(i), loop[0])]))
    out = mesh.copy()
    out.faces = np.vstack(new_faces)
    return out


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

    right_q, left_q = QUADRANTS[jaw]
    right = [c for f, c in centroids.items() if quadrant(f) == right_q]
    left = [c for f, c in centroids.items() if quadrant(f) == left_q]
    if not right or not left:
        raise ValueError("需要左右两侧都有牙齿才能估计正中矢状面")
    # 左右方向和正中矢状面：优先只用左右两侧都存在的同名牙对（每对“左 − 右”取平均）。
    # 直接用两侧全部牙的质心之差，一侧缺牙时两侧牙数不对称，方向会偏转好几度，
    # 镜像供体和镜像位置在牙弓末端就会偏出几毫米
    pairs = [(c, centroids[mirror(f)]) for f, c in centroids.items() if quadrant(f) == right_q and mirror(f) in centroids]
    if len(pairs) >= 2:
        lateral = np.mean([b - a for a, b in pairs], axis=0)
        origin = np.mean([(a + b) / 2 for a, b in pairs], axis=0)
    else:
        lateral = np.mean(left, axis=0) - np.mean(right, axis=0)
        origin = (np.mean(left, axis=0) + np.mean(right, axis=0)) / 2
    lateral -= (lateral @ normal) * normal
    lateral /= np.linalg.norm(lateral)
    return ArchFrame(origin=origin, occlusal=normal, lateral=lateral)
