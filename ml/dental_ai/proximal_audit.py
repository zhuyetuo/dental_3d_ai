"""Optional geometric review of proximal contacts on closed generated crowns.

Measurements use the supplied coordinate system in millimeters. The depth and
area thresholds flag cases for research review.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .meshdist import closest_point

SEEDS = (0, 3, 11)
N_SAMPLES = 16000
EDGE_MARGIN_MM = .18
DEPTH_MM = .15
AREA_MM2 = .02


def _border_tree(surface: trimesh.Trimesh) -> cKDTree | None:
    boundary = surface.edges_unique[np.bincount(surface.edges_unique_inverse) == 1]
    if not len(boundary):
        return None
    edges = np.asarray(surface.vertices)[boundary]
    t = np.linspace(0., 1., 7)[None, :, None]
    points = edges[:, :1] * (1 - t) + edges[:, 1:] * t
    return cKDTree(points.reshape(-1, 3))


def _inside_samples(crown, surface, up, floor, seed, depth_threshold, border):
    points = trimesh.sample.sample_surface(surface, N_SAMPLES, seed=seed)[0]
    low, high = crown.bounds
    in_bounds = ((points >= low) & (points <= high)).all(axis=1)
    near = points[in_bounds & (points @ up >= floor)]
    if not len(near):
        return 0., 0., 0.
    closest, distance, _, normals = closest_point(crown, near, return_normal=True)
    inward = (np.einsum('ij,ij->i', near - closest, normals) < 0) & (distance > .035)
    points, depths = near[inward], distance[inward]
    if not len(points):
        return 0., 0., 0.
    inside = crown.contains(points)
    points, depths = points[inside], depths[inside]
    if not len(points):
        return 0., 0., 0.
    near_edge = border.query(points)[0] <= EDGE_MARGIN_MM if border is not None else np.zeros(len(points), bool)
    interior = depths[~near_edge]
    sample_area = float(surface.area) / N_SAMPLES
    return (float(interior.max(initial=0.)),
            float(sample_area * (interior > depth_threshold).sum()),
            float(sample_area * (depths[near_edge] > .05).sum()))


def _segment_hits(a, b, up, floor):
    edges = np.asarray(a.vertices)[a.edges_unique]
    mins, maxs = edges.min(axis=1), edges.max(axis=1)
    low, high = b.bounds
    relevant = ((maxs >= low) & (mins <= high)).all(axis=1) & ((edges @ up).max(axis=1) >= floor)
    tree = b.triangles_tree
    candidates = [list(tree.intersection(tuple(lo) + tuple(hi)))
                  for lo, hi in zip(mins[relevant] - 1e-8, maxs[relevant] + 1e-8)]
    lengths = np.array([len(indices) for indices in candidates], dtype=np.intp)
    if not lengths.sum():
        return np.empty((0, 3))
    starts, ends = edges[relevant][np.repeat(np.arange(len(lengths)), lengths)].transpose(1, 0, 2)
    indices = np.fromiter((i for part in candidates for i in part), dtype=np.intp)
    v0, v1, v2 = np.asarray(b.triangles)[indices].transpose(1, 0, 2)
    ray, e1, e2 = ends - starts, v1 - v0, v2 - v0
    h = np.cross(ray, e2)
    determinant = np.einsum('ij,ij->i', e1, h)
    offset = starts - v0
    q = np.cross(offset, e1)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.einsum('ij,ij->i', offset, h) / determinant
        v = np.einsum('ij,ij->i', ray, q) / determinant
        t = np.einsum('ij,ij->i', e2, q) / determinant
    hits = (np.abs(determinant) > 1e-10) & (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
    hits &= (t >= -1e-7) & (t <= 1 + 1e-7)
    points = starts[hits] + t[hits, None] * ray[hits]
    return points[points @ up >= floor]


def _surface_crossings(crown, neighbor, up, floor, border):
    points = np.vstack((_segment_hits(crown, neighbor, up, floor), _segment_hits(neighbor, crown, up, floor)))
    if not len(points):
        return 0, 0
    _, selected = np.unique(np.round(points / .015).astype(np.intp), axis=0, return_index=True)
    points = points[selected]
    edge = int((border.query(points)[0] <= EDGE_MARGIN_MM).sum()) if border is not None else 0
    return edge, len(points) - edge


def audit_proximal(crown, neighbor, up, floor, legacy_gap=None) -> dict:
    """Report repeated interior depth, area and boundary crossing evidence."""
    if not crown.is_watertight or crown.volume <= 0:
        return {'review_status': 'unmeasurable_mesh'}
    border = _border_tree(neighbor)
    values = np.array([_inside_samples(crown, neighbor, up, floor, seed, DEPTH_MM, border) for seed in SEEDS])
    depth = float(np.min(values[:, 0]))
    area = float(np.min(values[:, 1]))
    boundary_area = float(np.min(values[:, 2]))
    edge_hits, interior_hits = _surface_crossings(crown, neighbor, up, floor, border)
    if depth > DEPTH_MM and area > AREA_MM2:
        status = 'supported_penetration'
    elif interior_hits >= 3:
        status = 'interior_crossing_below_depth_area_gate'
    elif edge_hits >= 3 or boundary_area > AREA_MM2:
        status = 'open_boundary_crossing'
    elif legacy_gap is not None and legacy_gap < -.15:
        status = 'legacy_flag_without_geometric_support'
    else:
        status = 'no_supported_penetration'
    return dict(review_status=status, interior_depth_mm=depth, interior_area_gt_015_mm2=area,
                boundary_area_gt_005_mm2=boundary_area, boundary_crossings=edge_hits,
                interior_crossings=interior_hits)
