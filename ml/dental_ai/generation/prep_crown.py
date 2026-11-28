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
from ..meshdist import closest_point
from ..preferences import DesignPreferences
from ..geometry import clean_crown, tooth_mesh
from .base import GenerationContext
from .contacts import adjust_contacts
from .occlusion import adjust_occlusion
from .mirror_template import MirrorTemplateGenerator


@dataclass
class PrepCrown:
    fdi: int
    mesh: trimesh.Trimesh
    info: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    die: trimesh.Trimesh | None = None  # 颈缘线以内的预备体表面（网页上看贴合用）


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
    margin_h = _margin_floor(margin, frame, theta)
    f = margin_field(xy, h, poly, inside_2d, margin_h)
    cut, values = split_along_zero(local, f)
    inner = (values <= 0)[cut.faces].all(axis=1)
    if not inner.any():
        raise ValueError("颈缘线以内没有预备体表面，请检查颈缘线")
    die = cut.submesh([np.nonzero(inner)[0]], append=True, repair=False)
    die.merge_vertices()
    # 只留最大连通块，补上扫描本身的小洞（真实口扫上一颗预备体有几百个）。不补的话组织面有洞，
    # 牙冠不封闭；内外判断在洞附近也不可靠，最小厚度的迭代会把外表面越推越远
    return clean_crown(die)


def _margin_floor(margin: np.ndarray, frame: _Frame, theta: np.ndarray, bins: int = 72, window: int = 3) -> np.ndarray:
    """各方位附近（±window 个 5° 格子）颈缘线的最低高度。

    切割时用来排除颈缘线以下的部分（如肩台下方、多边形内侧的牙龈）。不能直接用同一方位的颈缘线高度：
    真实颈缘线随牙龈呈扇贝形，邻面处 2–3 mm 的升降只占很小的角度，颈缘线内侧的肩台点按方位
    对到的却是更高的那段颈缘线，会被误判为“在颈缘线以下”而切掉，切出来的边界满是向内的尖刺。"""
    t_m, _, h_m = frame.polar(margin)
    k = np.floor((t_m + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    low = np.full(bins, np.inf)
    np.minimum.at(low, k, h_m)
    idx = np.arange(bins)
    ok = np.isfinite(low)
    low = np.interp(idx, idx[ok], low[ok], period=bins)
    low = np.min([np.roll(low, s) for s in range(-window, window + 1)], axis=0)
    x = (theta + np.pi) / (2 * np.pi) * bins - 0.5
    return np.interp(x, idx, low, period=bins)


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


def make_intaglio(
    die: trimesh.Trimesh, margin: np.ndarray, frame: _Frame, cement_gap: float, gap_taper: float, block_out: bool, max_blockout: float = 2.0
):
    """冠内表面顶点：外偏粘接间隙，去倒凹。返回 (顶点, 去倒凹的最大外推量)。"""
    d = _distance_to_loop(die.vertices, margin)
    weight = np.clip(d / gap_taper, 0.0, 1.0)
    weight = weight * weight * (3 - 2 * weight)  # smoothstep
    verts = die.vertices + die.vertex_normals * (cement_gap * weight)[:, None]
    shift_max = 0.0
    if block_out:
        # 倒凹 = 从就位方向（正上方）看过去，被预备体自身挡住的点。只填补这些点：
        # 沿径向推到正上方挡住它的那一点的半径（填平悬突下面的凹陷），最多 max_blockout。
        # 不能推到“同一方位更高处的最大半径”：中央窝里的点（半径很小）也可能被窝沟壁挡住，
        # 那样会被推出 4–5 mm 到轴壁外，组织面整个变形（真实口扫上出现过）
        probe = trimesh.Trimesh(verts, die.faces, process=False)
        origins = verts + frame.up * 1e-3
        dirs = np.tile(frame.up, (len(verts), 1))
        occluded = probe.ray.intersects_any(origins, dirs)
        occluded &= weight > 0.0  # 颈缘线本身不动
        if occluded.any():
            idx = np.nonzero(occluded)[0]
            hits, ray, _ = probe.ray.intersects_location(origins[idx], dirs[idx], multiple_hits=False)
            _, r, _ = frame.polar(verts[idx[ray]])
            _, r_hit, _ = frame.polar(hits)
            push = np.zeros(len(verts))
            push[idx[ray]] = np.clip(r_hit - r, 0.0, max_blockout)
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

    old = np.asarray(outer.vertices, float).copy()
    theta, r, h = frame.polar(old)
    hb, rb, ht, rt = at(h_b, theta), at(r_b, theta), at(h_t, theta), at(r_t, theta)
    s = np.clip((h - hb) / band, 0.0, 1.0)  # 0 = 底边，1 = 不受影响
    new_h = np.where(s < 1.0, ht + s * (hb + band - ht), h)
    new_r = r + (1.0 - s) * (rt - rb)
    d = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    new = (
        frame.center
        + (new_r * d[:, 0])[:, None] * frame.u
        + (new_r * d[:, 1])[:, None] * frame.v
        + new_h[:, None] * frame.up
    )
    new[loop] = target + frame.radial(target) * thickness
    # 每个方位角单独算的位移在颈缘线陡升陡降处（邻面）相邻顶点差很多，颈部一圈被挤出尖棱（网页上看是侧面毛刺）：
    # 位移场在网格上平滑（底边固定在颈缘线上、带以外不动）。46 号示例：这一步的翻折 120 → 见 _smooth_field
    disp = new - old
    fixed = np.zeros(len(old), bool)
    fixed[loop] = True
    disp = _smooth_field(outer, disp, fixed | (s >= 1.0), iters=20)
    outer.vertices = old + disp
    outer.vertices[loop] = new[loop]


def _side_bulge(outer: trimesh.Trimesh, loop: np.ndarray, margin: np.ndarray, frame: _Frame, offsets) -> None:
    """侧面按相对高度（颈缘线 0 → 最高点 1）沿法向外加厚 offsets = (下, 中, 上) mm（三段中点 1/6、1/2、5/6 之间线性插值）。
    评分卡量到生成的牙冠侧面普遍比原牙瘦 0.2–0.35 mm（形状模型 + 扩散这条路）。只动侧面（法向和咬合方向夹角大），
    颈缘线 1.5 mm 内渐变到 0，位移在网格上平滑。"""
    off = np.asarray(offsets, float)
    if not np.any(off):
        return
    v = np.asarray(outer.vertices, float)
    theta, _, h = frame.polar(v)
    tm, _, hm_loop = frame.polar(margin)
    o = np.argsort(tm)
    hm = np.interp(theta, np.concatenate([tm[o] - 2 * np.pi, tm[o], tm[o] + 2 * np.pi]), np.tile(hm_loop[o], 3))
    top = float(np.percentile(h, 99.5))
    t = np.clip((h - hm) / np.maximum(top - hm, 1e-6), 0, 1)
    delta = np.interp(t, [1 / 6, 1 / 2, 5 / 6], off)
    n = outer.vertex_normals
    side = np.clip((0.75 - np.abs(n @ frame.up)) / 0.35, 0, 1)
    ramp = np.clip((_distance_to_loop(v, margin) - 0.3) / 1.2, 0, 1)
    disp = n * (delta * side * ramp)[:, None]
    fixed = np.zeros(len(v), bool)
    fixed[loop] = True
    outer.vertices = v + _smooth_field(outer, disp, fixed, iters=6)


def _smooth_field(mesh: trimesh.Trimesh, field: np.ndarray, fixed: np.ndarray, iters: int = 10) -> np.ndarray:
    """网格上平滑一个逐顶点的量（均匀拉普拉斯，fixed 的顶点保持原值）。"""
    e = mesh.edges_unique
    n = len(mesh.vertices)
    deg = np.maximum(np.bincount(e.ravel(), minlength=n), 1).astype(float)
    f = np.asarray(field, float).copy()
    two_d = f.ndim == 1
    if two_d:
        f = f[:, None]
    keep = f[fixed].copy()
    for _ in range(iters):
        acc = np.empty_like(f)
        for k in range(f.shape[1]):
            acc[:, k] = np.bincount(e[:, 0], f[e[:, 1], k], n) + np.bincount(e[:, 1], f[e[:, 0], k], n)
        f = 0.5 * f + 0.5 * acc / deg[:, None]
        f[fixed] = keep
    return f[:, 0] if two_d else f


def _ring_max(mesh: trimesh.Trimesh, values: np.ndarray, rings: int = 2) -> np.ndarray:
    """逐顶点的量在邻近 rings 圈内取最大（灰度膨胀）。"""
    e = mesh.edges_unique
    v = np.asarray(values, float).copy()
    for _ in range(rings):
        m = v.copy()
        np.maximum.at(m, e[:, 0], v[e[:, 1]])
        np.maximum.at(m, e[:, 1], v[e[:, 0]])
        v = m
    return v


def _local_taubin(mesh: trimesh.Trimesh, movable: np.ndarray, rounds: int = 6, ring: int = 3, iters: int = 6, thr: float = 0.0) -> int:
    """皱褶处局部 Taubin 平滑（λ 0.5 / μ −0.53，不缩体积）：只动相邻面夹角大（法向点积 < thr，默认 > 90°）的地方周围几圈的可动顶点。
    切向摊平只能把翻过来的三角形摊开，挤皱的地方（法向来回翻但没翻过来）摊不平。原地修改，返回剩下的尖棱数。"""
    v = np.asarray(mesh.vertices, float).copy()
    f = np.asarray(mesh.faces)
    e = mesh.edges_unique
    n = len(v)
    deg = np.maximum(np.bincount(e.ravel(), minlength=n), 1).astype(float)[:, None]
    left = 0
    for _ in range(rounds):
        m = trimesh.Trimesh(v, f, process=False)
        fn = m.face_normals
        adj = m.face_adjacency
        bad = (fn[adj[:, 0]] * fn[adj[:, 1]]).sum(1) < thr
        left = int(bad.sum())
        if not left:
            break
        act = np.zeros(n, bool)
        act[f[adj[bad].ravel()].ravel()] = True
        for _ in range(ring):
            a2 = act.copy()
            a2[e[act[e[:, 0]], 1]] = True
            a2[e[act[e[:, 1]], 0]] = True
            act = a2
        act &= movable
        w = act[:, None].astype(float)
        for _ in range(iters):
            for s in (0.5, -0.53):
                acc = np.empty_like(v)
                for k in range(3):
                    acc[:, k] = np.bincount(e[:, 0], v[e[:, 1], k], n) + np.bincount(e[:, 1], v[e[:, 0], k], n)
                v = v + s * (acc / deg - v) * w
    mesh.vertices = v
    return left


def _unfold(mesh: trimesh.Trimesh, movable: np.ndarray, rounds: int = 15, ring: int = 4, inner: int = 10) -> int:
    """局部摊平翻折处（相邻面夹角 > 120°）：只动翻折处周围几圈的可动顶点，切向拉普拉斯（沿当前法向去掉法向分量，不改形状）。
    原地修改，返回剩下的翻折边数。"""
    v = np.asarray(mesh.vertices, float).copy()
    e = mesh.edges_unique
    n = len(v)
    deg = np.maximum(np.bincount(e.ravel(), minlength=n), 1).astype(float)[:, None]
    left = 0
    for _ in range(rounds):
        m = trimesh.Trimesh(v, mesh.faces, process=False)
        fn = m.face_normals
        adj = m.face_adjacency
        bad = (fn[adj[:, 0]] * fn[adj[:, 1]]).sum(1) < -0.5
        left = int(bad.sum())
        if not left:
            break
        act = np.zeros(n, bool)
        act[np.asarray(mesh.faces)[adj[bad].ravel()].ravel()] = True
        for _ in range(ring):
            a2 = act.copy()
            a2[e[act[e[:, 0]], 1]] = True
            a2[e[act[e[:, 1]], 0]] = True
            act = a2
        act &= movable
        w = act[:, None].astype(float)
        for _ in range(inner):
            nrm = trimesh.Trimesh(v, mesh.faces, process=False).vertex_normals
            acc = np.empty_like(v)
            for k in range(3):
                acc[:, k] = np.bincount(e[:, 0], v[e[:, 1], k], n) + np.bincount(e[:, 1], v[e[:, 0], k], n)
            lap = acc / deg - v
            v = v + 0.5 * (lap - (lap * nrm).sum(1, keepdims=True) * nrm) * w
    mesh.vertices = v
    return left


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


def _signed_thickness(points: np.ndarray, inner: trimesh.Trimesh, edge_margin: float = 0.3, return_normals: bool = False):
    """外表面点到组织面的有符号距离（组织面法向一侧为正）。

    最近点落在组织面开放边缘（颈缘）edge_margin 以内的点记为 +inf：那里用边缘三角形的法向判断内外不可靠，
    会被误判成“太薄”，最小厚度的迭代就把这些点一次次往外推（真实口扫上推出了 10 mm 以上）。"""
    cp, _, tri = closest_point(inner, points)
    t = np.einsum("ij,ij->i", points - cp, inner.face_normals[tri])
    loops = boundary_loops(inner)
    if loops:
        near_edge = cKDTree(inner.vertices[loops[0]]).query(cp)[0] < edge_margin
        t[near_edge] = np.inf
    return (t, inner.face_normals[tri]) if return_normals else t


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
    library=None,
    outer_generator=None,
) -> PrepCrown:
    prefs = preferences or DesignPreferences()
    # 颈缘线的点先吸附回网格表面：检测结果经过平滑，在凸棱处会稍微缩进牙体，
    # 人工编辑的点也可能略微离开表面；底边沿着离开表面的线走，牙冠就会切进肩台
    margin = closest_point(mesh, np.asarray(margin, dtype=float))[0]
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
    if target is None or (target.mesial is None and target.distal is None):
        raise ValueError(f"{fdi} 两侧都没有邻牙，无法确定牙冠方向和大小")
    ctx = GenerationContext(
        mesh=mesh, labels=without, jaw=jaw, antagonist=antagonist, excluded_donors=frozenset(excluded_donors)
    )
    generator = MirrorTemplateGenerator(template_dir=template_dir, preferences=prefs, library=library)
    outer_tooth = None
    if outer_generator is not None:  # 流程的生成器（形状模型 + 扩散补细节）：把预备体当缺失牙生成外形
        try:
            if getattr(outer_generator, "accepts_prep", False):  # 能看着颈缘线和预备体表面生成的形状模型
                cand = outer_generator.generate(ctx, target, margin=crown_margin, prep=die)
            elif getattr(outer_generator, "accepts_margin", False):  # 能看着颈缘线生成的（颈缘线条件扩散模型）
                cand = outer_generator.generate(ctx, target, margin=crown_margin)
            else:
                cand = outer_generator.generate(ctx, target)
            if len(boundary_loops(cand.crown)) == 1:
                outer_tooth = cand
            else:
                warnings.append(f"{fdi} 生成器的外表面边界不是一圈，改用镜像模板")
        except (LookupError, ValueError) as exc:
            warnings.append(f"{fdi} 生成器没能生成外表面（{exc}），改用镜像模板")
    if outer_tooth is None:
        outer_tooth = generator.generate(ctx, target, center=margin.mean(axis=0))
    outer = outer_tooth.crown.copy()
    outer_loops = boundary_loops(outer)
    if len(outer_loops) != 1:
        raise ValueError(f"外表面应只有一个开放边界，实际 {len(outer_loops)} 个")
    outer_loop = outer_loops[0]
    _morph_outer(outer, outer_loop, crown_margin, frame, margin_thickness)
    import os

    # 侧面加厚（下 / 中 / 上，mm）：评分卡偶数号病人量到侧面比原牙瘦 0.13 / 0.20 / 0.23 mm，补回这么多；
    # 奇数号病人验证：瘦 0.17 / 0.20 / 0.25 → 0.07 / 0.09 / 0.16 mm，外形 RMSE 0.459 → 0.435
    side_off = [float(x) for x in os.environ.get("PREP_SIDE_OFFSET", "0.13,0.20,0.23").split(",")]
    if not str(outer_tooth.info.get("source", "")).count("voxel"):  # 16 / 26 体素模型这条路侧面不瘦（评分卡 +0.0–0.5 mm），不加
        _side_bulge(outer, outer_loop, crown_margin, frame, side_off)

    # 3. 邻接 + 最小厚度（颈缘线附近 gap_taper + 0.5 mm 内的顶点不动、不检查）
    dmarg = _distance_to_loop(outer.vertices, margin)
    far = dmarg > gap_taper + 0.5
    before = _signed_thickness(outer.vertices[far], inner) if far.any() else np.array([np.inf])
    neighbors = {f: tooth_mesh(mesh, without, f) for f in (target.mesial, target.distal) if f is not None}

    def enforce_thickness():
        for _ in range(20):
            t, normals = _signed_thickness(outer.vertices[far], inner, return_normals=True)
            # 每次最多外推一个冠厚；沿组织面在最近点处的法向推（一定让厚度增加）。
            # 用外表面自己的顶点法向不行：折叠、补洞处法向朝里，越推越薄，真实口扫上推出了 10 mm
            deficit = np.clip(prefs.crown_thickness - t, 0.0, prefs.crown_thickness)
            if deficit.max() <= 1e-4:
                break
            # 逐顶点各推各的会推出尖刺（侧面毛刺）：推的量先在周围 2 圈取最大、再平滑，推出来是一块平滑的鼓包
            push = np.zeros((len(outer.vertices), 3))
            push[far] = normals * deficit[:, None]
            mag = np.zeros(len(outer.vertices))
            mag[far] = deficit
            mag = _ring_max(outer, mag, 2)
            dirs = _smooth_field(outer, push, ~far, iters=10)
            dn = np.linalg.norm(dirs, axis=1, keepdims=True)
            dirs = np.where(dn > 1e-9, dirs / np.maximum(dn, 1e-9), 0.0)
            amount = _smooth_field(outer, mag, ~far, iters=6)
            full = np.zeros(len(outer.vertices))
            full[far] = deficit
            amount = np.maximum(amount, full)  # 平滑不能把该推的量抹小（评分卡上最小厚度掉到 0.07 mm）
            outer.vertices[far] = outer.vertices[far] + (dirs * amount[:, None])[far]
        # 平滑推法在个别地方方向会互相抵消（16 号示例最薄 −1.2 mm 推不出来）：还不够厚的点按原来的办法逐点推，保证不比原来差
        for _ in range(20):
            t, normals = _signed_thickness(outer.vertices[far], inner, return_normals=True)
            deficit = np.clip(prefs.crown_thickness - t, 0.0, prefs.crown_thickness)
            if deficit.max() <= 1e-4:
                break
            outer.vertices[far] = outer.vertices[far] + normals * deficit[:, None]

    # 补冠厚、邻接都会在弯处挤出皱褶（46 号示例补冠厚一步 +150 处尖棱，邻面角上看得见）：每一步之后皱褶处局部 Taubin 平滑
    tidy = os.environ.get("PREP_TIDY", "taubin")

    def smooth_creases():
        if tidy == "taubin":
            _local_taubin(outer, far)
        elif tidy == "unfold":
            _unfold(outer, far)

    enforce_thickness()
    smooth_creases()
    occlusion: dict = {}
    if antagonist is not None:
        occlusion, occlusion_warnings = adjust_occlusion(outer, antagonist, np.asarray(up, dtype=float), prefs.occlusion_distance, far)
        warnings += [f"{fdi} {w}" for w in occlusion_warnings]
    # 邻接不做平滑：前磨牙邻面颈缘线高，接触点就在分界上方 1–2 mm，平滑会把推力抹掉（评分卡上邻接间隙变成 +0.2 mm）
    contacts, contact_warnings = adjust_contacts(outer, neighbors, prefs.contacts_distance, far)
    smooth_creases()
    enforce_thickness()  # 退开邻牙、平滑时可能变薄，再补一次厚度
    # 平滑会把邻接处刚推出来的小鼓包也抹掉（评分卡上近中没接触的牙 21% → 45%）：再调一次邻接
    contacts, contact_warnings = adjust_contacts(outer, neighbors, prefs.contacts_distance, far)
    smooth_creases()
    enforce_thickness()  # 第二次退开邻牙、平滑也可能变薄（评分卡上个别磨牙最薄 0.12 mm）
    if antagonist is not None:  # 邻接、补厚度都可能又顶到对颌：最后再磨一遍（只磨不抬），和缺失牙流程一样
        again, _ = adjust_occlusion(outer, antagonist, np.asarray(up, dtype=float), prefs.occlusion_distance, far, lift=False)
        occlusion["occlusal_clearance_mm"] = again.get("occlusal_clearance_mm")
        occlusion["occlusal_reduction_mm"] = round(occlusion.get("occlusal_reduction_mm", 0.0) + again.get("occlusal_reduction_mm", 0.0), 3)
    after = _signed_thickness(outer.vertices[far], inner) if far.any() else np.array([np.inf])
    if np.isfinite(after).any() and float(np.min(after)) < prefs.crown_thickness - 0.05:
        warnings.append(f"{fdi} 咬合空间不够：最薄处 {float(np.min(after)):.2f} mm，低于设定的 {prefs.crown_thickness:.2f} mm（预备体咬合面要多磨一些）")
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
        **occlusion,
    }
    return PrepCrown(fdi=fdi, mesh=crown, info=info, warnings=warnings, die=die)
