import numpy as np
from scipy import ndimage

CELL = 0.15


def groove(points):
    y = points[:, 1]
    p = points[y >= np.percentile(y, 98) - 4.0]
    x, z, h = p[:, 0], p[:, 2], p[:, 1]
    ix, iz = np.floor((x - x.min()) / CELL).astype(int), np.floor((z - z.min()) / CELL).astype(int)
    g = np.full((ix.max() + 1, iz.max() + 1), -np.inf)
    np.maximum.at(g, (ix, iz), h)
    have = np.isfinite(g)
    idx = ndimage.distance_transform_edt(~have, return_distances=False, return_indices=True)
    gf = ndimage.gaussian_filter(g[tuple(idx)], 0.5)
    depth = ndimage.grey_closing(gf, size=(9, 9)) - gf
    edge = ndimage.distance_transform_edt(ndimage.binary_closing(have, iterations=2)) * CELL
    d = depth[have & (edge >= 1.0)]
    return float(np.percentile(d, 90)), float(d.mean())
