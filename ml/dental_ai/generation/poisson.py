"""带法向的点 → 光滑表面网格：谱方法泊松重建（Shape As Points 的 DPSR）+ 表面网格提取（surface nets），全在 GPU 上，不需要额外的库。

    verts, faces = poisson_mesh(points, normals, res=192, sigma=1.0, trim=0.3)

- 泊松：法向三线性撒到网格上得向量场 V，FFT 里解 ∇²χ = ∇·V，再乘高斯低通（sigma，单位：格）；
- 等值面取输入点处 χ 的平均；
- surface nets：每个跨零的格子放一个顶点（各条边上过零点的平均），每条跨零的格边连成两个三角形；
- trim：离输入点超过 trim mm 的面去掉（牙冠外表面是开口的，泊松会在底下补一个盖子），只留最大的连通块。
"""
from __future__ import annotations

import math

import numpy as np


def _splat(g, vals, res):
    import torch

    i0 = torch.floor(g).long()
    f = g - i0
    out = torch.zeros(vals.shape[1], res ** 3, device=g.device, dtype=vals.dtype)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                idx = (i0 + torch.tensor([dx, dy, dz], device=g.device)).clamp(0, res - 1)
                flat = (idx[:, 0] * res + idx[:, 1]) * res + idx[:, 2]
                out.index_add_(1, flat, (vals * w[:, None]).T)
    return out.view(-1, res, res, res)


def _sample(grid, g):
    """三线性取值，g 是格坐标 (N,3)。"""
    import torch

    res = grid.shape[0]
    i0 = torch.floor(g).long()
    f = g - i0
    v = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                idx = (i0 + torch.tensor([dx, dy, dz], device=g.device)).clamp(0, res - 1)
                v = v + w * grid[idx[:, 0], idx[:, 1], idx[:, 2]]
    return v


def dpsr(points, normals, res: int = 192, sigma: float = 1.0, pad: float = 0.2, device: str = "cuda"):
    """返回 (chi 网格, 等值, 原点 lo, 格宽 h)。"""
    import torch

    P = torch.as_tensor(np.asarray(points), dtype=torch.float32, device=device)
    N = torch.as_tensor(np.asarray(normals), dtype=torch.float32, device=device)
    N = N / N.norm(dim=1, keepdim=True).clamp(min=1e-8)
    lo_p, hi_p = P.min(0)[0], P.max(0)[0]
    size = float((hi_p - lo_p).max()) * (1 + 2 * pad)
    h = size / res
    lo = (lo_p + hi_p) / 2 - size / 2
    g = (P - lo) / h
    V = _splat(g, N, res)
    Vh = torch.fft.fftn(V, dim=(1, 2, 3))
    k = torch.fft.fftfreq(res, device=device) * 2 * math.pi  # 弧度 / 格
    kx, ky, kz = torch.meshgrid(k, k, k, indexing="ij")
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    div = 1j * (kx * Vh[0] + ky * Vh[1] + kz * Vh[2])
    k2s = k2.clone()
    k2s[0, 0, 0] = 1
    chi_h = -div / k2s * torch.exp(-0.5 * sigma ** 2 * k2)
    chi_h[0, 0, 0] = 0
    chi = torch.fft.ifftn(chi_h).real
    iso = float(_sample(chi, g).mean())
    return chi, iso, lo, h


def surface_nets(s):
    """s：(R,R,R) 标量场（GPU 张量），零等值面 → (顶点 (V,3) 格坐标, 三角形 (F,3))，都是 numpy。"""
    import torch

    R = s.shape[0]
    dev = s.device
    neg = s < 0
    # 每个格子 8 个角
    corners = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if sum(abs(corners[i][t] - corners[j][t]) for t in range(3)) == 1]
    C = R - 1
    vals = torch.stack([s[a:a + C, b:b + C, c:c + C] for a, b, c in corners])  # (8,C,C,C)
    acc = torch.zeros(C, C, C, 3, device=dev)
    cnt = torch.zeros(C, C, C, device=dev)
    for i, j in edges:
        va, vb = vals[i], vals[j]
        cross = (va < 0) != (vb < 0)
        t = (va / (va - vb).where(cross, torch.ones_like(va))).clamp(0, 1)
        pa = torch.tensor(corners[i], device=dev, dtype=torch.float32)
        pb = torch.tensor(corners[j], device=dev, dtype=torch.float32)
        pos = pa + t[..., None] * (pb - pa)
        acc += pos * cross[..., None]
        cnt += cross
    active = cnt > 0
    vid = torch.full((C, C, C), -1, dtype=torch.long, device=dev)
    nv = int(active.sum())
    vid[active] = torch.arange(nv, device=dev)
    base = torch.stack(torch.meshgrid(*(torch.arange(C, device=dev),) * 3, indexing="ij"), -1).float()
    verts = (base + acc / cnt.clamp(min=1)[..., None])[active]
    faces = []
    for ax in range(3):
        b, c = (ax + 1) % 3, (ax + 2) % 3
        # 沿 ax 方向的格边：从格点 p 到 p+e_ax，跨零；周围 4 个格子：p 在 b、c 方向上 −1 / 0
        sl = [slice(0, R)] * 3
        sl[ax] = slice(0, R - 1)
        s0 = neg[tuple(sl)]
        sl2 = list(sl)
        sl2[ax] = slice(1, R)
        s1 = neg[tuple(sl2)]
        cross = s0 != s1
        # 只要 b、c 坐标都在 1..R-2（四个格子都存在）
        m = torch.zeros_like(cross)
        idx = [slice(None)] * 3
        idx[b] = slice(1, R - 1)
        idx[c] = slice(1, R - 1)
        m[tuple(idx)] = True
        cross = cross & m
        p = torch.nonzero(cross)
        flip = s0[cross]
        q = []
        for db, dc in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
            o = p.clone()
            o[:, b] += db
            o[:, c] += dc
            q.append(vid[o[:, 0], o[:, 1], o[:, 2]])
        q = torch.stack(q, 1)
        q = torch.where(flip[:, None], q, q.flip(1))
        faces.append(torch.cat([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]))
    f = torch.cat(faces)
    f = f[(f >= 0).all(1)]
    return verts.cpu().numpy(), f.cpu().numpy()


def poisson_mesh(points, normals, res: int = 192, sigma: float = 1.0, trim: float | None = 0.3, device: str = "cuda"):
    import trimesh
    from scipy.spatial import cKDTree

    chi, iso, lo, h = dpsr(points, normals, res, sigma, device=device)
    v, f = surface_nets(chi - iso)
    v = v * h + lo.cpu().numpy()
    m = trimesh.Trimesh(v, f, process=True)
    # 朝向：面法向和最近输入点的法向一致
    t = cKDTree(np.asarray(points))
    d, i = t.query(m.triangles_center)
    if (np.asarray(m.face_normals) * np.asarray(normals)[i]).sum(1).mean() < 0:
        m.invert()
    if trim:
        keep = d < trim
        m.update_faces(keep)
        m.remove_unreferenced_vertices()
        parts = m.split(only_watertight=False)
        if len(parts):
            m = max(parts, key=lambda x: x.area)
    return m


def deepen_grooves(m, k: float = 2.0, up_axis: int = 1, cell: float = 0.1, size: int = 13, blend: float = 1.0):
    """只加深窄沟：咬合面高度图（cell mm 一格）做 size 格的形态学闭运算（把比约 1.3 mm 窄的沟填平）减原高度 = 沟深，
    朝咬合面的顶点沿 −up 再挪 k × 沟深。平面、牙尖、大的窝不变（闭运算后不变），所以不放大噪声。
    用在泊松表面上（三角形均匀，不会折叠），再把牙冠网格投上去。坐标里 up_axis 轴朝咬合面。"""
    import trimesh
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from trimesh.grouping import group_rows

    if k <= 0:
        return m
    ax = [i for i in range(3) if i != up_axis]
    v = np.asarray(m.vertices, float).copy()
    pts, _ = trimesh.sample.sample_surface(m, 150000, seed=5)
    y = pts[:, up_axis]
    top0 = np.percentile(y, 98) - 4.0
    p = pts[y >= top0]
    x0, z0 = p[:, ax[0]].min(), p[:, ax[1]].min()
    ix = np.floor((p[:, ax[0]] - x0) / cell).astype(int)
    iz = np.floor((p[:, ax[1]] - z0) / cell).astype(int)
    g = np.full((ix.max() + 1, iz.max() + 1), -np.inf)
    np.maximum.at(g, (ix, iz), p[:, up_axis])
    have = np.isfinite(g)
    idx = ndimage.distance_transform_edt(~have, return_distances=False, return_indices=True)
    gf = ndimage.gaussian_filter(g[tuple(idx)], 0.75)
    depth = ndimage.gaussian_filter(ndimage.grey_closing(gf, size=(size, size)) - gf, 1.5)
    edge = ndimage.distance_transform_edt(ndimage.binary_closing(have, iterations=3)) * cell
    depth *= np.clip(edge / 1.0, 0, 1)  # 轮廓附近（边缘嵴外侧）不动
    n = np.asarray(m.vertex_normals)
    fx, fz = (v[:, ax[0]] - x0) / cell - 0.5, (v[:, ax[1]] - z0) / cell - 0.5
    dv = ndimage.map_coordinates(depth, [fx, fz], order=1, mode="nearest")
    hv = ndimage.map_coordinates(gf, [fx, fz], order=1, mode="nearest")
    w = np.clip((n[:, up_axis] - 0.3) / 0.4, 0, 1) * (v[:, up_axis] >= top0) * np.clip(1 - (np.abs(v[:, up_axis] - hv) - 0.2) / 0.2, 0, 1)
    es = m.edges_sorted
    rim = np.unique(es[group_rows(es, require_count=1)])
    if len(rim):
        w *= np.clip(cKDTree(v[rim]).query(v)[0] / blend, 0, 1)
    disp = k * dv * w
    e = m.edges_unique
    deg = np.bincount(e.ravel(), minlength=len(v)).astype(float)
    for _ in range(4):  # 位移在网格上平滑几次
        acc = np.bincount(e[:, 0], disp[e[:, 1]], len(v)) + np.bincount(e[:, 1], disp[e[:, 0]], len(v))
        disp = 0.5 * disp + 0.5 * acc / np.maximum(deg, 1)
    v[:, up_axis] -= disp
    return trimesh.Trimesh(v, m.faces, process=False)
