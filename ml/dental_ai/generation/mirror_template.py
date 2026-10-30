"""V1 生成方法：对侧同名牙镜像 + 缺隙拟合；没有对侧牙时使用模板库。

步骤：
1. 取供体牙：对侧同名牙（缺 36 取 46），否则从模板库读取同号牙 STL；
2. 关于正中矢状面镜像（模板库的牙已经是目标侧，不需要镜像）；
3. 平移到缺隙中心：水平位置取两侧邻牙最近点的中点，高度沿用供体牙；
4. 沿牙弓切线方向缩放，使与两侧邻牙的距离等于 contacts_distance（缩放限制在 0.8~1.2 倍以内）；
5. 若提供对颌牙，沿咬合方向下移，保证与对颌之间至少留 occlusion_distance 的间隙。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ..detection.missing import MissingTooth
from ..fdi import mirror
from ..geometry import ArchFrame, estimate_arch_frame, tooth_mesh
from ..io import load_mesh
from ..preferences import DesignPreferences
from .base import GeneratedTooth, GenerationContext, ToothGenerator


class MirrorTemplateGenerator(ToothGenerator):
    name = "mirror_template"

    def __init__(
        self,
        template_dir: str | Path | None = None,
        preferences: DesignPreferences | None = None,
        scale_limits: tuple[float, float] = (0.8, 1.2),
    ):
        self.template_dir = Path(template_dir) if template_dir else None
        self.prefs = preferences or DesignPreferences()
        self.scale_limits = scale_limits

    def _donor(self, ctx: GenerationContext, frame: ArchFrame, fdi: int) -> tuple[trimesh.Trimesh, str]:
        contra = mirror(fdi)
        if np.any(ctx.labels == contra):
            donor = tooth_mesh(ctx.mesh, ctx.labels, contra)
            donor.vertices = frame.reflect(donor.vertices)
            donor.invert()  # 镜像会翻转三角形朝向
            return donor, f"mirror:{contra}"
        if self.template_dir and (self.template_dir / f"{fdi}.stl").exists():
            return load_mesh(self.template_dir / f"{fdi}.stl"), f"template:{fdi}"
        raise LookupError(f"没有对侧牙 {contra}，模板库中也没有 {fdi}")

    def generate(self, ctx: GenerationContext, target: MissingTooth) -> GeneratedTooth:
        if target.kind != "space":
            raise ValueError(f"{target.fdi} 的缺隙类型为 {target.kind}，V1 只处理两侧都有邻牙的缺隙")
        frame = estimate_arch_frame(ctx.mesh, ctx.labels, ctx.jaw)
        crown, source = self._donor(ctx, frame, target.fdi)
        up = frame.occlusal

        a = ctx.mesh.vertices[ctx.labels == target.mesial]
        b = ctx.mesh.vertices[ctx.labels == target.distal]
        # 两侧邻牙互相最近的点
        da, ia = cKDTree(a).query(b, k=1)
        pb = b[np.argmin(da)]
        pa = a[ia[np.argmin(da)]]
        gap_center = (pa + pb) / 2

        tangent = pb - pa
        tangent -= (tangent @ up) * up
        tangent /= np.linalg.norm(tangent)

        # 平移：水平到缺隙中心，高度保持供体牙原有高度
        c = crown.vertices.mean(axis=0)
        shift = gap_center - c
        shift -= (shift @ up) * up
        crown.vertices = crown.vertices + shift
        c = crown.vertices.mean(axis=0)

        # 沿切线缩放
        width = np.ptp((crown.vertices - c) @ tangent)
        gap = float(target.gap_mm)
        scale = float(np.clip((gap - 2 * self.prefs.contacts_distance) / width, *self.scale_limits))
        along = ((crown.vertices - c) @ tangent)[:, None] * tangent
        crown.vertices = crown.vertices + along * (scale - 1.0)

        drop = 0.0
        if ctx.antagonist is not None:
            drop = self._occlusal_drop(crown, ctx.antagonist, up)
            crown.vertices = crown.vertices - up * drop

        return GeneratedTooth(
            fdi=target.fdi,
            crown=crown,
            method=self.name,
            info={"source": source, "scale": scale, "gap_mm": gap, "occlusal_drop_mm": drop},
        )

    def _occlusal_drop(self, crown: trimesh.Trimesh, antagonist: trimesh.Trimesh, up: np.ndarray) -> float:
        """沿咬合方向发射射线，求需要下移多少才能与对颌保持间隙。

        射线起点放在顶点下方 5 mm，这样已经穿入对颌的顶点也能得到负的距离。
        """
        origins = crown.vertices - up * 5.0
        dirs = np.tile(up, (len(origins), 1))
        locs, idx, _ = antagonist.ray.intersects_location(origins, dirs, multiple_hits=False)
        if len(idx) == 0:
            return 0.0
        dist = (locs - crown.vertices[idx]) @ up
        needed = self.prefs.occlusion_distance - dist.min()
        return float(max(needed, 0.0))
