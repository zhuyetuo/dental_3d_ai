"""预备体牙冠生成：冠内表面贴合预备体，冠外表面是牙齿形态，两者在颈缘线处缝合成封闭实体。

步骤：
1. 冠内表面（组织面）：取颈缘线以内的预备体表面，沿法向外偏粘接间隙（cement_gap）。
   间隙在离颈缘线 gap_taper mm 内从 0 渐变到设定值，保证颈缘处密合。预备体表面沿颈缘线精确切开
   （margin/cut.py），组织面的边界就是牙冠的实际颈缘，落在颈缘线上；
   然后沿就位方向（咬合方向）去除倒凹：从正上方看被预备体自身挡住的点，沿径向推到上方悬突的半径，
   否则牙冠戴不进去；
2. 冠外表面：把预备体当作“缺失牙”，复用 MirrorTemplateGenerator（对侧同名牙镜像、按邻牙缺隙缩放、
   有对颌时满足咬合距离）；再把外表面底部按高度单调重映射，让底边落到颈缘线外 margin_thickness mm 处；
3. 邻接与最小厚度：调整外表面，使与近远中邻牙的距离等于邻接距离（contacts.py）；
   外表面离冠内表面不足 crown_thickness 的地方（颈缘附近除外），沿法向向外推；
4. 缝合：外表面底边与冠内表面底边按绕牙轴的角度对齐，生成一圈三角带，输出封闭网格。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from ..detection.missing import detect_missing
from ..fdi import Jaw
from ..margin.cervical import boundary_loops
from ..margin.cut import margin_field, split_along_zero
from ..preferences import DesignPreferences
from ..geometry import tooth_mesh
from .base import GenerationContext
from .contacts import adjust_contacts
from .mirror_template import MirrorTemplateGenerator


@dataclass
class PrepCrown:
    fdi: int
    mesh: trimesh.Trimesh
    info: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Frame:
    center: np.ndarray
    up: np.ndarray
    u: np.ndarray
    v: np.ndarray

    def polar(self, pts: np.ndarray):
        d = pts - self.center
        x, y = d @ self.u, d @ self.v
        return np.arctan2(y, x), np.hypot(x, y), d @ self.up

    def radial(self, pts: np.ndarray) -> np.ndarray:
        d = pts - self.center
        d = d - (d @ self.up)[:, None] * self.up
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return d / np.maximum(n, 1e-9)


def _frame(margin: np.ndarray, up: np.ndarray) -> _Frame:
    up = up / np.linalg.norm(up)
    u = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(up, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    return _Frame(center=margin.mean(axis=0), up=up, u=u, v=np.cross(up, u))


def _inside_polygon(px: np.ndarray, py: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """射线法判断点是否在多边形内（向量化）。"""
    inside = np.zeros(len(px), dtype=bool)
    x0, y0 = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    for a, b, c, d in zip(x0, y0, x1, y1):
        crosses = (b > py) != (d > py)
        xint = (c - a) * (py - b) / np.where(d - b == 0, 1e-12, d - b) + a
        inside ^= crosses & (px < xint)
    return inside


def extract_die(mesh: trimesh.Trimesh, margin: np.ndarray, frame: _Frame) -> trimesh.Trimesh:
    """沿颈缘线把网格精确切开，取内侧的预备体表面（最大连通块）。边界正好落在颈缘线上。"""
    _, _, mh = frame.polar(margin)
    poly = np.stack([(margin - frame.center) @ frame.u, (margin - frame.center) @ frame.v], axis=1)
    # 先只取颈缘线附近和以内的部分，减少切割的计算量
    d = mesh.vertices - frame.center
    above = d @ frame.up >= mh.min() - 1.0
    near = _inside_polygon(d @ frame.u, d @ frame.v, poly) | (_distance_to_loop(mesh.vertices, margin) < 1.5)
    keep = (above & near)[mesh.faces].all(axis=1)
    if not keep.any():
        raise ValueError("颈缘线以内没有预备体表面，请检查颈缘线")
    local = mesh.submesh([np.nonzero(keep)[0]], append=True, repair=False)
    ld = local.vertices - frame.center
    xy = np.stack([ld @ frame.u, ld @ frame.v], axis=1)
    inside_2d = _inside_polygon(xy[:, 0], xy[:, 1], poly)
    theta, _, h = frame.polar(local.vertices)
    margin_h = frame.polar(_interp_loop(margin, frame, theta))[2]
    f = margin_field(xy, h, poly, inside_2d, margin_h)
    cut, values = split_along_zero(local, f)
    inner = (values <= 0)[cut.faces].all(axis=1)
    if not inner.any():
        raise ValueError("颈缘线以内没有预备体表面，请检查颈缘线")
    die = cut.submesh([np.nonzero(inner)[0]], append=True, repair=False)
    die.merge_vertices()
    parts = die.split(only_watertight=False)
    return max(parts, key=lambda p: len(p.faces)) if len(parts) > 1 else die


def _loop_by_angle(points: np.ndarray, frame: _Frame) -> tuple[np.ndarray, np.ndarray]:
    """按绕牙轴逆时针、从角度最小处开始排列闭合环，返回 (下标顺序, 展开后的角度)。"""
    theta, _, _ = frame.polar(points)
    # 环本身是有序的，只需要统一方向和起点
    unwrapped = np.unwrap(theta)
    order = np.arange(len(points))
    if unwrapped[-1] < unwrapped[0]:
        order = order[::-1]
    start = int(np.argmin(np.mod(theta[order], 2 * np.pi)))
    order = np.roll(order, -start)
    ang = np.unwrap(theta[order])
    ang = ang - ang[0] + np.mod(theta[order[0]], 2 * np.pi)
    return order, ang


def _interp_loop(points: np.ndarray, frame: _Frame, query_theta: np.ndarray) -> np.ndarray:
    """按角度在闭合环上插值。"""
    order, ang = _loop_by_angle(points, frame)
    p = points[order]
    ang = np.concatenate([ang, [ang[0] + 2 * np.pi]])
    p = np.vstack([p, p[:1]])
    q = np.mod(query_theta - ang[0], 2 * np.pi) + ang[0]
    return np.stack([np.interp(q, ang, p[:, k]) for k in range(3)], axis=1)


def _distance_to_loop(points: np.ndarray, loop: np.ndarray) -> np.ndarray:
    dense = np.vstack([loop[i] + (loop[(i + 1) % len(loop)] - loop[i]) * t for t in np.linspace(0, 1, 5, endpoint=False) for i in range(len(loop))])
    return cKDTree(dense).query(points)[0]


def make_intaglio(die: trimesh.Trimesh, margin: np.ndarray, frame: _Frame, cement_gap: float, gap_taper: float, block_out: bool):
    """冠内表面顶点：外偏粘接间隙，去倒凹。返回 (顶点, 去倒凹的最大外推量)。"""
    d = _distance_to_loop(die.vertices, margin)
    weight = np.clip(d / gap_taper, 0.0, 1.0)
    weight = weight * weight * (3 - 2 * weight)  # smoothstep
    verts = die.vertices + die.vertex_normals * (cement_gap * weight)[:, None]
    shift_max = 0.0
    if block_out:
        # 倒凹 = 从就位方向（正上方）看过去，被预备体自身挡住的点。只填补这些点：
        # 把它们沿径向推到同一方位、更高处的最大半径（填平悬突下面的凹陷）
        probe = trimesh.Trimesh(verts, die.faces, process=False)
        origins = verts + frame.up * 1e-3
        occluded = probe.ray.intersects_any(origins, np.tile(frame.up, (len(verts), 1)))
        occluded &= weight > 0.0  # 颈缘线本身不动
        if occluded.any():
            theta, r, h = frame.polar(verts)
            sectors = np.floor((theta + np.pi) / (2 * np.pi) * 72).astype(int) % 72
            push = np.zeros(len(verts))
            for s in np.unique(sectors[occluded]):
                idx = np.nonzero(sectors == s)[0]
                idx = idx[np.argsort(-h[idx])]  # 从咬合面往颈部
                need = np.maximum.accumulate(r[idx]) - r[idx]
                push[idx] = np.where(occluded[idx], need, 0.0)
            verts = verts + frame.radial(verts) * push[:, None]
            shift_max = float(push.max())
    return verts, shift_max


def _morph_outer(outer: trimesh.Trimesh, loop: np.ndarray, margin: np.ndarray, frame: _Frame, thickness: float, band: float = 2.5) -> None:
    """把外表面的底边移到颈缘线外 thickness mm 处，用按高度单调的重映射，保证不折叠。

    在每个方位 θ 上：供体底边高度 h_b(θ)、半径 r_b(θ)；目标是颈缘线的 h_m(θ)、r_m(θ) + thickness。
    底边以上 band mm 内的点，高度从 [h_b, h_b + band] 线性映射到 [h_m, h_b + band]，
    半径按同样的比例从“底边的偏移量”过渡到 0。按距离衰减直接拖动底边，会让底边上方的点
    比新底边还低，外表面在颈部向下折一下，钻进肩台。"""
    theta_b, r_b, h_b = frame.polar(outer.vertices[loop])
    order = np.argsort(theta_b)
    tb = np.concatenate([theta_b[order] - 2 * np.pi, theta_b[order], theta_b[order] + 2 * np.pi])

    def at(values: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return np.interp(theta, tb, np.tile(values[order], 3))

    target = _interp_loop(margin, frame, theta_b)
    _, r_t, h_t = frame.polar(target)
    r_t = r_t + thickness
    band = max(band, float((h_t - h_b).max()) + 1.5)

    theta, r, h = frame.polar(outer.vertices)
    hb, rb, ht, rt = at(h_b, theta), at(r_b, theta), at(h_t, theta), at(r_t, theta)
    s = np.clip((h - hb) / band, 0.0, 1.0)  # 0 = 底边，1 = 不受影响
    new_h = np.where(s < 1.0, ht + s * (hb + band - ht), h)
    new_r = r + (1.0 - s) * (rt - rb)
    d = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    outer.vertices = (
        frame.center
        + (new_r * d[:, 0])[:, None] * frame.u
        + (new_r * d[:, 1])[:, None] * frame.v
        + new_h[:, None] * frame.up
    )
    outer.vertices[loop] = target + frame.radial(target) * thickness


def _loft(a_idx: np.ndarray, a_ang: np.ndarray, b_idx: np.ndarray, b_ang: np.ndarray) -> np.ndarray:
    """两个按角度排好的闭合环之间的三角带（按角度交替推进）。"""
    faces = []
    i = j = 0
    na, nb = len(a_idx), len(b_idx)
    a_next = np.append(a_ang[1:], a_ang[0] + 2 * np.pi)
    b_next = np.append(b_ang[1:], b_ang[0] + 2 * np.pi)
    while i < na or j < nb:
        ai, bj = a_idx[i % na], b_idx[j % nb]
        if j >= nb or (i < na and a_next[i] <= b_next[j]):
            faces.append([ai, a_idx[(i + 1) % na], bj])
            i += 1
        else:
            faces.append([ai, b_idx[(j + 1) % nb], bj])
            j += 1
    return np.array(faces)


def _signed_thickness(points: np.ndarray, inner: trimesh.Trimesh) -> np.ndarray:
    cp, _, tri = trimesh.proximity.closest_point(inner, points)
    return np.einsum("ij,ij->i", points - cp, inner.face_normals[tri])


def generate_prep_crown(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    jaw: Jaw,
    fdi: int,
    margin: np.ndarray,
    up: np.ndarray,
    preferences: DesignPreferences | None = None,
    antagonist: trimesh.Trimesh | None = None,
    template_dir=None,
    gap_taper: float = 1.0,
    margin_thickness: float = 0.05,
    excluded_donors: set[int] | frozenset[int] = frozenset(),
) -> PrepCrown:
    prefs = preferences or DesignPreferences()
    # 颈缘线的点先吸附回网格表面：检测结果经过平滑，在凸棱处会稍微缩进牙体，
    # 人工编辑的点也可能略微离开表面；底边沿着离开表面的线走，牙冠就会切进肩台
    margin = trimesh.proximity.closest_point(mesh, np.asarray(margin, dtype=float))[0]
    frame = _frame(margin, np.asarray(up, dtype=float))
    warnings: list[str] = []

    # 1. 冠内表面
    die = extract_die(mesh, margin, frame)
    if len(die.faces) < 20:
        raise ValueError(f"{fdi} 颈缘线以内几乎没有预备体表面，请检查颈缘线")
    inner_verts, blockout = make_intaglio(die, margin, frame, prefs.cement_gap, gap_taper, prefs.undercuts == "remove")
    inner = trimesh.Trimesh(inner_verts, die.faces, process=False)
    inner_loops = boundary_loops(inner)
    if not inner_loops:
        raise ValueError(f"{fdi} 的预备体表面没有开放边界")
    inner_loop = inner_loops[0]
    # 不拖动组织面：网格已经沿颈缘线切开，边界就在颈缘线上、也在原始牙面上。
    # （把组织面拉到颈缘线上会把附近的组织面拖到肩台下面，牙冠就穿进预备体了。）
    # 牙冠的实际颈缘 = 组织面边界（粘接间隙在这里为 0，与预备体表面重合）
    crown_margin = inner.vertices[inner_loop]

    # 2. 冠外表面：把预备体当作缺失牙生成
    without = np.where(labels == fdi, 0, labels)
    target = next((m for m in detect_missing(mesh, without, jaw, include_third_molars=True) if m.fdi == fdi), None)
    if target is None or target.kind != "space":
        raise ValueError(f"{fdi} 两侧没有邻牙，V1 无法确定牙冠大小")
    ctx = GenerationContext(
        mesh=mesh, labels=without, jaw=jaw, antagonist=antagonist, excluded_donors=frozenset(excluded_donors)
    )
    outer_tooth = MirrorTemplateGenerator(template_dir=template_dir, preferences=prefs).generate(ctx, target)
    outer = outer_tooth.crown
    outer_loops = boundary_loops(outer)
    if len(outer_loops) != 1:
        raise ValueError(f"外表面应只有一个开放边界，实际 {len(outer_loops)} 个")
    outer_loop = outer_loops[0]
    _morph_outer(outer, outer_loop, crown_margin, frame, margin_thickness)

    # 3. 邻接 + 最小厚度（颈缘线附近 gap_taper + 0.5 mm 内的顶点不动、不检查）
    far = _distance_to_loop(outer.vertices, margin) > gap_taper + 0.5
    before = _signed_thickness(outer.vertices[far], inner) if far.any() else np.array([np.inf])
    neighbors = {f: tooth_mesh(mesh, without, f) for f in (target.mesial, target.distal) if f is not None}

    def enforce_thickness():
        for _ in range(20):
            t = _signed_thickness(outer.vertices[far], inner)
            deficit = np.clip(prefs.crown_thickness - t, 0.0, None)
            if deficit.max() <= 1e-4:
                break
            outer.vertices[far] = outer.vertices[far] + outer.vertex_normals[far] * deficit[:, None]

    enforce_thickness()
    contacts, contact_warnings = adjust_contacts(outer, neighbors, prefs.contacts_distance, far)
    enforce_thickness()  # 退开邻牙时可能变薄，再补一次厚度
    after = _signed_thickness(outer.vertices[far], inner) if far.any() else np.array([np.inf])
    warnings += contact_warnings

    # 4. 缝合：外表面法向朝外，组织面翻转（实体的外法向在组织面上指向预备体）
    inner.invert()
    n_outer = len(outer.vertices)
    o_order, o_ang = _loop_by_angle(outer.vertices[outer_loop], frame)
    i_order, i_ang = _loop_by_angle(inner.vertices[inner_loop], frame)
    strip = _loft(outer_loop[o_order], o_ang, inner_loop[i_order] + n_outer, i_ang)
    crown = trimesh.Trimesh(
        np.vstack([outer.vertices, inner.vertices]),
        np.vstack([outer.faces, inner.faces + n_outer, strip]),
        process=False,
    )
    trimesh.repair.fix_normals(crown, multibody=False)
    if crown.volume < 0:
        crown.invert()
    if not crown.is_watertight:
        warnings.append(f"{fdi} 牙冠网格不封闭，请检查颈缘线")
    if blockout > 0.2:
        warnings.append(f"{fdi} 预备体有倒凹，组织面最多填补了 {blockout:.2f} mm")

    info = {
        "type": "prep_crown",
        "source": outer_tooth.info.get("source"),
        "scale": outer_tooth.info.get("scale"),
        "cement_gap_mm": prefs.cement_gap,
        "min_thickness_before_mm": round(float(before.min()), 3),
        "min_thickness_mm": round(float(after.min()), 3),
        "blockout_max_mm": round(blockout, 3),
        "volume_mm3": round(float(crown.volume), 2),
        "margin_points": int(len(crown_margin)),
        "contacts_mm": {str(k): v for k, v in contacts.items()},
    }
    return PrepCrown(fdi=fdi, mesh=crown, info=info, warnings=warnings)
