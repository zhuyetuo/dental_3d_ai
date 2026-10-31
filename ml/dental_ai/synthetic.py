"""合成牙弓：用于测试和前后端联调，在拿到真实口扫数据之前也能跑通整个流程。

网格是一张高度场：牙龈是平缓的马蹄形隆起，每颗牙是牙弓抛物线上的一个凸包，
牙齿区域与牙龈区域共享边，因此每颗牙都有真实的开放边界（牙颈线）。
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .fdi import Jaw, arch_order, expected_width


def synthetic_arch(
    jaw: Jaw = "lower",
    missing: tuple[int, ...] = (),
    resolution: float = 0.35,
    curvature: float = 0.045,
    band: float | None = None,
    prepared: tuple[int, ...] = (),
    label_offset: float = 0.0,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """返回 (网格, 逐顶点 FDI 标签)。

    坐标：单位 mm，牙冠朝 +Z，+Y 指向后方（磨牙），+X 为患者左侧。下颌与 Teeth3DS 标准坐标系一致；
    Teeth3DS 中上颌牙冠朝 -Z，需要时把上颌关于 XY 平面镜像（见 tests 中的 canonical_upper）。
    band：只保留离牙弓曲线 band mm 以内的部分，得到真实口扫那样的马蹄形轮廓；None 表示保留整块矩形。
    prepared：做成肩台预备体的牙号。预备体是“平顶 + 陡峭轴壁 + 0.8 mm 宽的水平肩台”，
        颈缘线就是肩台外缘（一个圆），真值记在 mesh.metadata["margins"][fdi] = (cx, cy, 半径)。
    label_offset：预备体的分割标签比真实颈缘线向外多出（>0）或向内缩进（<0）多少 mm，
        用来模拟分割边界的误差。
    """
    order = arch_order(jaw)
    widths = np.array([expected_width(f) for f in order])
    # 沿抛物线 y = curvature * x^2 按弧长排牙，中线位于两颗中切牙之间
    xs = np.linspace(-40, 40, 4001)
    ys = curvature * xs**2
    arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    arc -= np.interp(0, xs, arc)
    edges = np.concatenate([[0], np.cumsum(widths)])
    edges -= edges[len(order) // 2]
    centers_s = (edges[:-1] + edges[1:]) / 2
    cx = np.interp(centers_s, arc, xs)
    cy = curvature * cx**2

    gx = np.arange(cx.min() - 8, cx.max() + 8, resolution)
    gy = np.arange(-8, cy.max() + 8, resolution)
    X, Y = np.meshgrid(gx, gy)

    # 点到牙弓曲线的距离（离散近似）
    curve = cKDTree(np.stack([xs, ys], axis=1))
    d_curve = curve.query(np.stack([X.ravel(), Y.ravel()], axis=1))[0].reshape(X.shape)
    Z = 4.0 * np.exp(-(d_curve**2) / (2 * 4.0**2))  # 牙龈嵴

    labels = np.zeros(X.shape, dtype=np.int64)
    best = np.zeros(X.shape)
    margins = {}
    for f, x0, y0, w in zip(order, cx, cy, widths):
        if f in missing:
            continue
        r = w * 0.45
        rr = np.hypot(X - x0, Y - y0)
        if f in prepared:
            r_margin = 0.8 * r
            r_wall = r_margin - 0.8  # 肩台宽 0.8 mm
            r_top = r_wall - 0.5
            h = np.where(rr <= r_top, 4.5, np.where(rr <= r_wall, 4.5 - 3.5 * (rr - r_top) / (r_wall - r_top), 1.0))
            h = np.where(rr <= r_margin, h, 0.0)
            margins[f] = (float(x0), float(y0), float(r_margin))
            best = np.where(h > best, h, best)
            labels[rr <= r_margin + label_offset] = f
            continue
        bump = np.clip(1 - rr**2 / r**2, 0, None)
        h = 7.0 * np.sqrt(bump)
        inside = (h > 0) & (h > best)
        best = np.where(inside, h, best)
        labels[inside] = f
    Z = Z + best

    ny, nx = X.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    faces = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    mesh = trimesh.Trimesh(verts, faces, process=False)
    mesh.metadata["margins"] = margins
    labels = labels.ravel()
    if band is not None:
        keep = (d_curve.ravel() <= band)[faces].all(axis=1)
        used = np.unique(faces[keep])
        remap = np.full(len(verts), -1)
        remap[used] = np.arange(len(used))
        mesh = trimesh.Trimesh(verts[used], remap[faces[keep]], process=False)
        mesh.metadata["margins"] = margins
        labels = labels[used]
    return mesh, labels


def simulate_missing(mesh: trimesh.Trimesh, labels: np.ndarray, jaw: Jaw, teeth) -> tuple[trimesh.Trimesh, np.ndarray]:
    """在真实口扫上模拟缺牙：把牙齿顶点沿咬合方向压到周围一圈牙龈的高度（再低 0.3 mm），并标成牙龈。

    顶点和三角形都不删，顶点顺序不变，原来的真牙可以直接用作生成结果的对照。"""
    from .geometry import estimate_arch_frame

    mesh, labels = mesh.copy(), np.asarray(labels).copy()
    up = estimate_arch_frame(mesh, labels, jaw).occlusal
    v = mesh.vertices.copy()
    edges = mesh.edges_unique

    def flat(p):
        return p - np.outer(p @ up, up)

    for fdi in teeth:
        t = labels == fdi
        if not t.any():
            raise ValueError(f"网格中没有牙齿 {fdi}")
        cross = edges[t[edges[:, 0]] != t[edges[:, 1]]].ravel()
        ring = np.unique(cross[labels[cross] == 0])
        _, idx = cKDTree(flat(v[ring])).query(flat(v[t]), k=6)
        target = (v[ring] @ up)[idx].mean(axis=1) - 0.3
        v[t] += np.outer(target - v[t] @ up, up)
        labels[t] = 0
    mesh.vertices = v
    return mesh, labels


def simulate_preparation(
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    jaw: Jaw,
    fdi: int,
    margin_above_gum: float = 0.8,
    axial: float = 1.0,
    occlusal: float = 1.5,
    taper_deg: float = 6.0,
    scallop: float = 0.4,
) -> tuple[trimesh.Trimesh, np.ndarray]:
    """在真实口扫上把一颗牙“磨”成肩台预备体，返回 (网格, 真值颈缘线上的点)。

    颈缘线放在牙颈线（低通平滑、起伏幅度乘 scallop）上方 margin_above_gum mm；颈缘线以上的部分沿水平方向向牙轴内收
    axial mm（越往上收得越多，形成 taper_deg 的聚合角），并整体压低 occlusal mm。跨过颈缘线的三角形
    被拉成近似水平的肩台，肩台外缘就是颈缘线。顶点、三角形和标签都不变（整颗牙仍标为 fdi）。
    返回的真值颈缘线是肩台外缘那一圈没有移动的顶点。"""
    from .geometry import estimate_arch_frame
    from .margin.cervical import cervical_line

    mesh = mesh.copy()
    up = estimate_arch_frame(mesh, labels, jaw).occlusal
    t = np.nonzero(labels == fdi)[0]
    v = mesh.vertices
    axis = v[t].mean(axis=0)
    u = np.cross(up, [1.0, 0.0, 0.0])
    u = u / np.linalg.norm(u)
    w = np.cross(up, u)

    def polar(p):
        d = p - axis
        return np.arctan2(d @ w, d @ u), d @ up

    # 牙颈线高度随方位角变化：取 4 阶以内的傅里叶分量（保留邻面处的扇贝形起伏，去掉锯齿）
    theta_c, h_c = polar(cervical_line(mesh, labels, fdi, smooth_iters=0))
    k = np.arange(5)
    basis = np.concatenate([np.cos(np.outer(theta_c, k)), np.sin(np.outer(theta_c, k[1:]))], axis=1)
    coef = np.linalg.lstsq(basis, h_c, rcond=None)[0]

    # 标签边界在邻面处一直升到接触点（高 3–4 mm），真实预备体的颈缘线只随牙龈起伏 1–2 mm：起伏幅度压到 40%
    floor = float((basis @ coef).min())

    def margin_height(theta):
        b = np.concatenate([np.cos(np.outer(theta, k)), np.sin(np.outer(theta, k[1:]))], axis=1)
        return floor + scallop * (b @ coef - floor) + margin_above_gum

    theta, h = polar(v[t])
    hm = margin_height(theta)
    above = h > hm
    idx = t[above]
    d = v[idx] - axis
    horiz = d - np.outer(d @ up, up)
    r = np.linalg.norm(horiz, axis=1)
    rise = h[above] - hm[above]
    # 颈缘线处（各方位）牙面的半径；轴壁从这里内收 axial，再按聚合角向上收拢。
    # 牙冠最凸处比颈部宽，只按比例内收会留下倒凹，真实预备体的轴壁不会有倒凹
    bins = 72
    near = np.abs(h - hm) < 0.3
    kb = np.floor((theta[near] + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    rd = np.linalg.norm((v[t][near] - axis) - np.outer((v[t][near] - axis) @ up, up), axis=1)
    r_margin = np.full(bins, np.nan)
    for b in range(bins):
        if (kb == b).any():
            r_margin[b] = rd[kb == b].max()
    ok = ~np.isnan(r_margin)
    r_margin = np.interp(np.arange(bins), np.nonzero(ok)[0], r_margin[ok], period=bins)
    x = (theta[above] + np.pi) / (2 * np.pi) * bins - 0.5
    wall = np.interp(x, np.arange(bins), r_margin, period=bins) - axial - rise * np.tan(np.radians(taper_deg))
    target = np.minimum(r - axial, wall)
    scale = np.maximum(target, 0.35 * r) / np.maximum(r, 1e-9)  # 细的切牙不要收过轴心
    top = rise.max()
    lowered = rise * max(top - occlusal, 0.5 * top) / top  # 高度按比例压低，颈缘线处不动
    # 邻面：口扫只拍到接触点，标签边界以下没有牙面。在那里照样内收会在预备体和牙龈之间拉出一道
    # 竖直的“悬崖”（大倒凹）。离非本牙顶点 1 mm 以内逐渐减小磨除量，邻面成为刃状边缘
    others = np.nonzero(labels != fdi)[0]
    dist = cKDTree(v[others]).query(v[idx])[0]
    weight = np.clip(dist / 1.0, 0.0, 1.0)
    weight = weight * weight * (3 - 2 * weight)
    scale = 1.0 - weight * (1.0 - scale)
    new = axis + horiz * scale[:, None] + np.outer(hm[above] + lowered, up)
    v[idx] = v[idx] + weight[:, None] * (new - v[idx])
    mesh.vertices = v

    # 真值颈缘线：颊舌侧是肩台外缘（与磨过的顶点相邻、没有动的本牙顶点），
    # 邻面是磨过的区域贴着非本牙顶点的那一圈（刃状边缘）
    moved = np.zeros(len(v), dtype=bool)
    moved[idx] = True
    e = mesh.edges_unique
    cross = e[moved[e[:, 0]] != moved[e[:, 1]]]
    ends = np.unique(cross.ravel())
    outer = ends[~moved[ends] & (labels[ends] == fdi)]
    rim = e[(labels[e[:, 0]] == fdi) != (labels[e[:, 1]] == fdi)].ravel()
    rim = np.unique(rim[moved[rim]])
    truth = np.concatenate([outer, rim])
    return mesh, mesh.vertices[truth]
