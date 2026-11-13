"""牙冠质量评分卡：一颗牙冠（天然牙或生成的）在邻接、咬合、形态三方面的量化指标。

只做测量，不做判断：及格线由天然牙在同一套指标上的分布给出（见 eval_crown.py）。所有距离的单位是 mm。

邻接（每侧邻牙一组，前缀 mesial_ / distal_）：
- gap：> 0 是牙冠表面到邻牙表面的最小距离（有缝）；< 0 是挤进邻牙的最大深度。天然牙和邻牙在口扫上连成一片，
  接触点就在两颗牙的分界线上，所以量出来接近 0；
- contact_area：离邻牙 0.1 mm 以内的牙冠表面积（mm²）。

咬合（给了对颌才有，前缀 occl_）：
- clearance：牙冠表面到对颌表面的最小有符号距离。> 0 没咬上，< 0 穿进了对颌；
- penetration_area：穿入对颌超过 0.05 mm 的牙冠表面积（mm²），生成的牙冠这一项应当接近 0；
- contact_area：离对颌 0.2 mm 以内的表面积（mm²）；
- contacts：接触区的个数（0.2 mm 以内的点按 0.6 mm 连成的簇）。

形态（不需要参考牙）：
- relief：咬合面的起伏，牙尖到窝沟的高度差（咬合面内圈高度的 95% 分位 − 5% 分位）；
- detail：咬合面的细节量，高度图减去它的平滑版本（σ = 0.8 mm）后的均方根。偏平滑的牙冠这一项低；
- height_vs_neighbors：牙冠最高处 − 两侧邻牙最高处的平均值，沿咬合方向；
- arch_offset：牙冠中心偏离两侧邻牙中心连线的水平距离（颊舌向错位）。

有参考牙（被去掉的真牙）时另外给出 eval_generation.compare 的形状误差。
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from .generation.contacts import signed_distance_to_surface
from .margin.cervical import boundary_loops
from .meshdist import closest_point

CONTACT_MM = 0.1  # 邻接：这么近算接触
OCCL_CONTACT_MM = 0.2  # 咬合：这么近算接触（口扫的咬合配准本身有 0.1 mm 量级的误差）
PENETRATION_MM = 0.05


def _sample(mesh: trimesh.Trimesh, n: int = 12000) -> np.ndarray:
    return np.asarray(trimesh.sample.sample_surface(mesh, n, seed=0)[0])


def _crop(surface: trimesh.Trimesh, points: np.ndarray, reach: float) -> trimesh.Trimesh | None:
    """只留 surface 上离 points 的包围盒 reach 以内的三角形（对颌是整个颌，只有牙冠对面那一小片有用）。"""
    lo, hi = points.min(axis=0) - reach, points.max(axis=0) + reach
    v = np.asarray(surface.vertices)
    inside = np.all((v >= lo) & (v <= hi), axis=1)
    faces = np.asarray(surface.faces)
    keep = inside[faces].all(axis=1)
    if keep.sum() < 10:
        return None
    return surface.submesh([np.flatnonzero(keep)], append=True, repair=False)


def proximal(points: np.ndarray, area: float, neighbor: trimesh.Trimesh, up: np.ndarray, band: float = 4.5, cell: float = 0.25) -> dict:
    """只看牙冠上部（最高处往下 band 以内）：邻接点都在这一段里，封闭的生成牙冠在牙龈下的底座不算。

    有缝时 gap = 两个表面的最小距离。挤进邻牙的深度不能靠“在邻牙表面的哪一侧”来判断：邻牙是按标签切出来的开放曲面，
    边界就在邻接点旁边，内外判断在那里不可靠（试过，会把贴得很好的牙冠判成穿入 1 mm 以上）。这里沿近远中方向
    逐条线比较：同一条线上，牙冠朝邻牙一侧的表面越过了邻牙朝牙冠一侧的表面多少。"""
    up = up / np.linalg.norm(up)
    n_all = len(points)
    points = points[points @ up >= np.percentile(points @ up, 98) - band]
    area = area * len(points) / n_all
    other = _sample(neighbor, 12000)
    dist = cKDTree(other).query(points)[0]
    close = np.flatnonzero(dist <= 1.0)  # 靠近邻牙的点换成到三角面的精确距离（到采样点的距离有 0.1 mm 量级的下限）
    if len(close):
        dist[close] = closest_point(neighbor, points[close])[1]
    axis = other.mean(axis=0) - points.mean(axis=0)
    axis -= (axis @ up) * up
    axis /= np.linalg.norm(axis)
    side = np.cross(up, axis)
    lo_h, lo_t = points @ up, points @ side
    h0, t0 = lo_h.min(), lo_t.min()
    shape = (int((lo_h.max() - h0) / cell) + 1, int((lo_t.max() - t0) / cell) + 1)

    def cells(p):
        ih, it = np.floor((p @ up - h0) / cell).astype(int), np.floor((p @ side - t0) / cell).astype(int)
        ok = (ih >= 0) & (ih < shape[0]) & (it >= 0) & (it < shape[1])
        return ih[ok], it[ok], (p @ axis)[ok]

    far = np.full(shape, -np.inf)  # 牙冠在每条线上朝邻牙一侧的表面
    ih, it, sc = cells(points)
    np.maximum.at(far, (ih, it), sc)
    near = np.full(shape, np.inf)  # 邻牙在每条线上朝牙冠一侧的表面（只算在牙冠中心外侧的点）
    ih, it, sn = cells(other)
    beyond = sn > float(np.median(points @ axis))
    np.minimum.at(near, (ih[beyond], it[beyond]), sn[beyond])
    both = np.isfinite(far) & np.isfinite(near)
    depth = float((far - near)[both].max()) if both.any() else 0.0
    return {"gap": -depth if depth > 0.03 else float(dist.min()), "contact_area": float((dist <= CONTACT_MM).mean() * area)}


def occlusion(points: np.ndarray, area: float, antagonist: trimesh.Trimesh) -> dict:
    near = _crop(antagonist, points, 4.0)
    if near is None:
        return {"clearance": None, "penetration_area": 0.0, "contact_area": 0.0, "contacts": 0}
    d, _ = signed_distance_to_surface(points, near, cutoff=3.0)
    ok = np.isfinite(d)
    if not ok.any():
        return {"clearance": None, "penetration_area": 0.0, "contact_area": 0.0, "contacts": 0}
    touching = ok & (d <= OCCL_CONTACT_MM)
    n_contacts = 0
    if touching.sum() >= 3:
        p = points[touching]
        pairs = cKDTree(p).query_pairs(0.6, output_type="ndarray")
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(p), len(p)))
        _, comp = connected_components(graph, directed=False)
        n_contacts = int((np.bincount(comp) >= 3).sum())
    return {
        "clearance": float(d[ok].min()),
        "penetration_area": float((ok & (d < -PENETRATION_MM)).mean() * area),
        "contact_area": float(touching.mean() * area),
        "contacts": n_contacts,
    }


def _frame(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = up / np.linalg.norm(up)
    a = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(a) < 0.3:
        a = np.cross(up, [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    return a, np.cross(up, a)


def occlusal_table(points: np.ndarray, up: np.ndarray, cell: float = 0.2, inset: float = 1.5, sigma: float = 0.8, depth: float = 4.0) -> dict:
    """咬合面的起伏和细节量。把牙冠上部（最高处往下 depth 以内，大致到外形高点）沿咬合方向投影成高度图（每格取最高点），
    只看离轮廓 inset 以内的内圈。只取上部是因为封闭的生成牙冠带着牙龈下的底座，天然牙没有，整颗投影两者的轮廓不可比。"""
    a, b = _frame(up)
    points = points[points @ up >= np.percentile(points @ up, 98) - depth]
    x, y, h = points @ a, points @ b, points @ up
    ix = np.floor((x - x.min()) / cell).astype(int)
    iy = np.floor((y - y.min()) / cell).astype(int)
    grid = np.full((ix.max() + 1, iy.max() + 1), -np.inf)
    np.maximum.at(grid, (ix, iy), h)
    filled = np.isfinite(grid)
    filled = ndimage.binary_closing(filled, iterations=2)
    inner = ndimage.distance_transform_edt(filled) * cell >= inset
    valid = inner & np.isfinite(grid)
    if valid.sum() < 30:
        return {"relief": None, "detail": None}
    z = np.where(valid, grid, 0.0)
    weight = ndimage.gaussian_filter(valid.astype(float), sigma / cell)
    smooth = ndimage.gaussian_filter(z, sigma / cell) / np.maximum(weight, 1e-6)
    heights = grid[valid]
    return {
        "relief": float(np.percentile(heights, 95) - np.percentile(heights, 5)),
        "detail": float(np.sqrt(np.mean((grid[valid] - smooth[valid]) ** 2))),
    }


def _top(points: np.ndarray, up: np.ndarray) -> float:
    return float(np.percentile(points @ up, 98))


def placement(points: np.ndarray, neighbor_points: list[np.ndarray], up: np.ndarray) -> dict:
    out = {"height_vs_neighbors": None, "arch_offset": None}
    if not neighbor_points:
        return out
    out["height_vs_neighbors"] = _top(points, up) - float(np.mean([_top(p, up) for p in neighbor_points]))
    if len(neighbor_points) == 2:
        up = up / np.linalg.norm(up)

        def flat(p):  # 牙冠上半部分的水平中心
            top = p[p @ up > np.percentile(p @ up, 50)]
            c = top.mean(axis=0)
            return c - (c @ up) * up

        c, n1, n2 = flat(points), flat(neighbor_points[0]), flat(neighbor_points[1])
        axis = (n2 - n1) / np.linalg.norm(n2 - n1)
        rel = c - n1
        out["arch_offset"] = float(np.linalg.norm(rel - (rel @ axis) * axis))
    return out


def score_crown(
    crown: trimesh.Trimesh,
    up: np.ndarray,
    mesial: trimesh.Trimesh | None = None,
    distal: trimesh.Trimesh | None = None,
    antagonist: trimesh.Trimesh | None = None,
    reference: trimesh.Trimesh | None = None,
) -> dict:
    """crown：牙冠（开放的牙冠面或封闭实体都可以）；up：咬合方向（指向对颌）。返回一行指标，缺的项为 None。"""
    points = _sample(crown)
    area = float(crown.area)
    row: dict = {"watertight": bool(crown.is_watertight)}
    neighbor_points = []
    for side, neighbor in (("mesial", mesial), ("distal", distal)):
        if neighbor is None or len(neighbor.faces) == 0:
            continue
        row.update({f"{side}_{k}": v for k, v in proximal(points, area, neighbor, up).items()})
        neighbor_points.append(_sample(neighbor, 4000))
    if antagonist is not None:
        row.update({f"occl_{k}": v for k, v in occlusion(points, area, antagonist).items()})
    # 高度图每格取最高点，点少了会抖（斜面上一格内高度就差 0.2 mm），封闭的生成牙冠还有一半的点落在底座上；
    # 这里单独密采样，保证天然牙和生成牙冠每格都有几十个点，细节量才可比
    row.update(occlusal_table(_sample(crown, 200000), up))
    row.update(placement(points, neighbor_points, up))
    if reference is not None:
        from .eval_generation import compare

        row.update({f"ref_{k}": v for k, v in compare(crown, reference, up).items()})
    return row
