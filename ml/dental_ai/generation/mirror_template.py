"""V1 生成方法：对侧同名牙镜像，没有可用的对侧牙时使用标准牙模板；再按邻牙拟合大小和位置。

步骤：
1. 放置位置：
   - 两侧都有邻牙（space）：两侧邻牙最近点的中点，切线方向由两点确定；
   - 只有一侧有邻牙（游离端，free_end）：对侧同名牙在时，取它关于正中矢状面的镜像位置；
     否则用邻牙和它旁边两颗牙拟合一段二次曲线沿牙弓外推（牙宽 = 标准宽度 × 整副牙列的比例）；
   - 调用方可以直接给定中心（预备体牙冠：颈缘线中心）；
2. 供体牙：对侧同名牙（缺 36 取 46，预备体不能当供体），关于正中矢状面镜像，高度沿用；
   否则从模板库取同牙位的模板（templates/），按目标位置的牙齿局部坐标系放到牙颈线高度；
3. 沿牙弓切线方向缩放到目标宽度（缩放限制在 0.8~1.2 倍以内；精确的邻接由 contacts.py 调整）；
4. 若提供对颌牙，沿咬合方向下移，保证与对颌之间至少留 occlusion_distance 的间隙。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ..detection.missing import MissingTooth
from ..fdi import arch_order, expected_width, mirror
from ..geometry import ArchFrame, estimate_arch_frame, tooth_mesh
from ..preferences import DesignPreferences
from ..templates.frame import build_frame, mesial_neighbors
from ..templates.library import TemplateLibrary
from .base import GeneratedTooth, GenerationContext, ToothGenerator


@dataclass
class Placement:
    center: np.ndarray  # 牙冠中心（水平位置）；高度为牙颈线高度
    mesial: np.ndarray  # 近中方向（水平单位向量）
    width: float  # 目标近远中宽度
    kind: str


def _cervical_height(mesh: trimesh.Trimesh, labels: np.ndarray, fdi: int, up: np.ndarray) -> float:
    from ..margin.cervical import cervical_line

    try:
        return float((cervical_line(mesh, labels, fdi, smooth_iters=0) @ up).mean())
    except ValueError:
        pts = mesh.vertices[labels == fdi] @ up
        return float(pts.min())


def _between(jaw, a: int, b: int) -> list[int]:
    """牙弓上严格位于 a、b 之间的牙位（两端都不含），按从 a 往 b 的顺序。"""
    order = arch_order(jaw, include_third_molars=True)
    i, j = order.index(a), order.index(b)
    return order[i + 1 : j] if j > i else order[j + 1 : i][::-1]


def _horizontal(v: np.ndarray, up: np.ndarray) -> np.ndarray:
    v = v - (v @ up) * up
    return v / max(np.linalg.norm(v), 1e-9)


def _extrapolate_arch(mesh: trimesh.Trimesh, labels: np.ndarray, jaw, neighbor: int, distance: float, up: np.ndarray, distal: bool):
    """从邻牙质心沿牙弓外推 distance（弧长，mm），返回 (位置, 该处的近中方向)。

    用邻牙和它另一侧最近的两颗现存牙的质心（投影到咬合平面），按弦长参数各拟合一段二次曲线再外推。
    不用整体牙弓坐标系：一侧缺牙时，按左右质心估计的左右方向会偏转，牙弓末端误差很大。"""
    chain = [neighbor]
    for _ in range(2):
        mes, dis = mesial_neighbors(labels, jaw, chain[-1])
        nxt = mes if distal else dis
        if nxt is None:
            break
        chain.append(nxt)
    pts = np.array([mesh.vertices[labels == f].mean(axis=0) for f in chain[::-1]])  # 从远离目标的一端到邻牙
    pts = pts - np.outer(pts @ up, up)
    t = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    deg = min(2, len(pts) - 1)
    if deg < 1:
        raise ValueError(f"{neighbor} 附近现存的牙太少，无法外推牙弓")
    coef = [np.polyfit(t, pts[:, k], deg) for k in range(3)]

    def at(x):
        return np.array([np.polyval(c, x) for c in coef])

    x, travelled, step = t[-1], 0.0, 0.02
    while travelled < distance:
        travelled += float(np.linalg.norm(at(x + step) - at(x)))
        x += step
    tangent = np.array([np.polyval(np.polyder(c), x) for c in coef])
    tangent = _horizontal(tangent, up)  # 指向远离牙弓中段（朝目标）的方向
    return at(x), -tangent


def _arch_curve_point(mesh: trimesh.Trimesh, labels: np.ndarray, jaw, mesial: int, distal: int, point: np.ndarray, up: np.ndarray) -> np.ndarray:
    """缺隙两侧各取最多两颗现存牙的质心，拟合局部二次曲线，返回曲线上与 point 沿牙弓位置相同的点。

    只用两颗邻牙质心的连线不行：前牙区牙弓弯曲大，31、41 都缺时连线比真实位置偏舌侧约 4 mm。"""
    order = arch_order(jaw, include_third_molars=True)
    present = {int(v) for v in np.unique(labels) if v != 0}
    i, j = sorted((order.index(mesial), order.index(distal)))
    left = [f for f in reversed(order[:i + 1]) if f in present][:2]
    right = [f for f in order[j:] if f in present][:2]
    pts = np.array([mesh.vertices[labels == f].mean(axis=0) for f in left[::-1] + right])
    ca, cb = mesh.vertices[labels == order[i]].mean(axis=0), mesh.vertices[labels == order[j]].mean(axis=0)
    e1 = _horizontal(cb - ca, up)
    e2 = np.cross(up, e1)
    x, y = (pts - ca) @ e1, (pts - ca) @ e2
    coef = np.polyfit(x, y, 2 if len(pts) >= 3 else 1)
    xp = (point - ca) @ e1
    return ca + xp * e1 + np.polyval(coef, xp) * e2


class MirrorTemplateGenerator(ToothGenerator):
    name = "mirror_template"

    def __init__(
        self,
        template_dir: str | Path | None = None,
        preferences: DesignPreferences | None = None,
        scale_limits: tuple[float, float] = (0.8, 1.2),
        library: TemplateLibrary | None = None,
    ):
        self.library = library or TemplateLibrary(template_dir)
        self.prefs = preferences or DesignPreferences()
        self.scale_limits = scale_limits

    def _arch_scale(self, ctx: GenerationContext, fdi: int, mesial_dir: np.ndarray, up: np.ndarray) -> float:
        """邻牙实测宽度 / 标准宽度，用来把标准牙宽换算到这副牙列。"""
        tangent = _horizontal(mesial_dir, up)
        pts = ctx.mesh.vertices[ctx.labels == fdi]
        return float(np.ptp(pts @ tangent)) / expected_width(fdi)

    def placement(self, ctx: GenerationContext, arch: ArchFrame, target: MissingTooth, center: np.ndarray | None = None) -> Placement:
        up = arch.occlusal
        mesh, labels = ctx.mesh, ctx.labels
        if target.mesial is not None and target.distal is not None:
            a = mesh.vertices[labels == target.mesial]
            b = mesh.vertices[labels == target.distal]
            da, ia = cKDTree(a).query(b, k=1)
            pb = b[np.argmin(da)]
            pa = a[ia[np.argmin(da)]]
            mesial_dir = _horizontal(pa - pb, up)
            gap = float(np.linalg.norm(pa - pb)) if target.gap_mm is None else float(target.gap_mm)
            # 一段连续缺失（如 35、36 都缺）：按各自标准宽度的比例分配缺隙
            run = _between(ctx.jaw, target.mesial, target.distal)
            total = sum(expected_width(f) for f in run)
            before = sum(expected_width(f) for f in run[: run.index(target.fdi)])
            share = gap / total
            offset = (before + expected_width(target.fdi) / 2) * share
            c = pa - mesial_dir * offset if center is None else np.asarray(center, float)
            contra = mirror(target.fdi)
            if center is None and len(run) > 1 and np.any(labels == contra):
                # 连续缺失、对侧同名牙在：用镜像位置，比按标准宽度比例分配准
                c = arch.reflect(mesh.vertices[labels == contra].mean(axis=0)[None])[0]
            elif center is None:
                # 沿牙弓方向用缺隙位置（真实口扫上误差 ≤ 0.4 mm），颊舌方向不能用：接触点不在牙冠颊舌向的中间，
                # 偏差 1–3 mm。颊舌位置取对侧同名牙的镜像位置，没有对侧牙时取缺隙两侧牙齿质心拟合的局部牙弓曲线
                if np.any(labels == contra):
                    ref = arch.reflect(mesh.vertices[labels == contra].mean(axis=0)[None])[0]
                else:
                    ref = _arch_curve_point(mesh, labels, ctx.jaw, target.mesial, target.distal, c, up)
                d = ref - c
                d -= (d @ up) * up + (d @ mesial_dir) * mesial_dir  # 只取水平面内垂直于牙弓的分量
                c = c + d
            height = (_cervical_height(mesh, labels, target.mesial, up) + _cervical_height(mesh, labels, target.distal, up)) / 2
            width = expected_width(target.fdi) * share - 2 * self.prefs.contacts_distance
            kind = "space"
        else:
            neighbor = target.mesial if target.mesial is not None else target.distal
            if neighbor is None:
                raise ValueError(f"{target.fdi} 两侧都没有邻牙")
            n_mes, _ = mesial_neighbors(labels, ctx.jaw, neighbor)
            cn = mesh.vertices[labels == neighbor].mean(axis=0)
            rough = _horizontal(mesh.vertices[labels == n_mes].mean(axis=0) - cn, up) if n_mes else -_horizontal(cn - arch.origin, up)
            scale = self._arch_scale(ctx, neighbor, rough, up)
            width = expected_width(target.fdi) * scale
            # 沿牙弓外推：从邻牙质心出发，走过邻牙的半个宽度、中间缺失的牙，再走半个目标牙宽
            toward_distal = target.mesial is not None
            neighbor_width = float(np.ptp(mesh.vertices[labels == neighbor] @ rough))
            skipped = sum(expected_width(f) for f in _between(ctx.jaw, neighbor, target.fdi)) * scale
            distance = neighbor_width / 2 + skipped + width / 2 - self.prefs.contacts_distance
            contra = mirror(target.fdi)
            if np.any(labels == contra):
                # 对侧同名牙在：用它关于正中矢状面的镜像位置（牙弓基本左右对称，比外推准）
                point = arch.reflect(mesh.vertices[labels == contra].mean(axis=0)[None])[0]
                mesial_dir = _horizontal(cn - point, up) if toward_distal else _horizontal(point - cn, up)
                placed_by = "mirror_position"
            else:
                point, mesial_dir = _extrapolate_arch(mesh, labels, ctx.jaw, neighbor, distance, up, distal=toward_distal)
                if not toward_distal:
                    mesial_dir = -mesial_dir
                placed_by = "arch_extrapolation"
            c = point if center is None else np.asarray(center, float)
            if center is not None:
                mesial_dir = _horizontal(cn - c, up) if toward_distal else _horizontal(c - cn, up)
            height = _cervical_height(mesh, labels, neighbor, up)
            kind = f"free_end:{placed_by}" if center is None else "free_end"
        c = c - (c @ up) * up + height * up
        return Placement(center=c, mesial=mesial_dir, width=width, kind=kind)

    def _donor(self, ctx: GenerationContext, arch: ArchFrame, fdi: int, place: Placement) -> tuple[trimesh.Trimesh, str]:
        contra = mirror(fdi)
        if contra not in ctx.excluded_donors and np.any(ctx.labels == contra):
            donor = tooth_mesh(ctx.mesh, ctx.labels, contra)
            donor.vertices = arch.reflect(donor.vertices)
            donor.invert()  # 镜像会翻转三角形朝向
            return donor, f"mirror:{contra}"
        template = self.library.get(fdi)
        frame = build_frame(arch, place.center, place.mesial)
        return frame.to_world(template.mesh), template.source

    def generate(self, ctx: GenerationContext, target: MissingTooth, center: np.ndarray | None = None) -> GeneratedTooth:
        """center：直接指定牙冠中心（预备体牙冠用颈缘线中心）。"""
        if target.kind == "closed":
            raise ValueError(f"{target.fdi} 的间隙已关闭，不生成")
        arch = estimate_arch_frame(ctx.mesh, ctx.labels, ctx.jaw)
        up = arch.occlusal
        place = self.placement(ctx, arch, target, center)
        crown, source = self._donor(ctx, arch, target.fdi, place)

        # 平移：水平到目标中心，高度保持（镜像牙沿用对侧高度，模板已放在牙颈线高度）
        c = crown.vertices.mean(axis=0)
        shift = place.center - c
        shift -= (shift @ up) * up
        crown.vertices = crown.vertices + shift
        c = crown.vertices.mean(axis=0)

        # 沿切线缩放到目标宽度
        tangent = place.mesial
        width = np.ptp((crown.vertices - c) @ tangent)
        scale = float(np.clip(place.width / width, *self.scale_limits))
        along = ((crown.vertices - c) @ tangent)[:, None] * tangent
        crown.vertices = crown.vertices + along * (scale - 1.0)

        height_info = {}
        if not source.startswith("mirror:"):
            height_info = self._fit_height(ctx, crown, target, place, up)

        drop = 0.0
        if ctx.antagonist is not None:
            drop = self._occlusal_drop(crown, ctx.antagonist, up)
            crown.vertices = crown.vertices - up * drop

        return GeneratedTooth(
            fdi=target.fdi,
            crown=crown,
            method=self.name,
            info={
                "source": source,
                "placement": place.kind,
                "scale": scale,
                "gap_mm": target.gap_mm,
                "target_width_mm": round(place.width, 2),
                "occlusal_drop_mm": drop,
                **height_info,
            },
        )

    def _fit_height(self, ctx: GenerationContext, crown: trimesh.Trimesh, target: MissingTooth, place: Placement, up: np.ndarray) -> dict:
        """模板牙（不是对侧镜像）的高度：牙尖顶对齐邻牙牙尖的平均高度（咬合平面），底边放在缺牙区牙龈上方，
        中间按比例竖直缩放。

        模板的尺寸是教科书平均值，原点放在邻牙牙颈线的平均高度；真实牙颈线在邻面处向上升高 2–4 mm，
        平均高度偏高，直接用会让模板牙比邻牙高出 4 mm 左右（真实口扫上测到的）。"""
        mesh, labels = ctx.mesh, ctx.labels
        tops = [float((mesh.vertices[labels == f] @ up).max()) for f in (target.mesial, target.distal) if f is not None and np.any(labels == f)]
        if not tops:
            return {}
        h = crown.vertices @ up
        top_now, bottom_now = float(h.max()), float(h.min())
        top = float(np.mean(tops))
        # 缺牙区牙龈高度：牙冠中心附近 3 mm 以内的牙龈顶点的中位数
        c = place.center
        gum = mesh.vertices[labels == 0]
        horiz = (gum - c) - np.outer((gum - c) @ up, up)
        near = np.linalg.norm(horiz, axis=1) < 3.0
        bottom = float(np.median(gum[near] @ up)) + 0.5 if near.any() else bottom_now
        s_z = float(np.clip((top - bottom) / max(top_now - bottom_now, 1e-6), 0.6, 1.3))
        new_h = top - (top_now - h) * s_z
        crown.vertices = crown.vertices + np.outer(new_h - h, up)
        return {"height_scale": round(s_z, 3), "top_aligned_to": [f for f in (target.mesial, target.distal) if f is not None]}

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
