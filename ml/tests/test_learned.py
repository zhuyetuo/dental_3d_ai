"""学习式生成（generation/learned.py）和学习式颈缘线（margin/learned.py）里不依赖训练好的权重的部分。"""

import numpy as np
import pytest
import trimesh

from dental_ai.generation import learned as gen
from dental_ai.margin.learned import zero_contour


def test_dome_template_is_open_with_one_boundary():
    verts, faces, boundary = gen.dome_template(rings=8, segments=16)
    mesh = trimesh.Trimesh(verts, faces, process=False)
    edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    border = edges[counts == 1]
    assert set(border.ravel()) == set(boundary.tolist())  # 只有最后一圈是开放边界
    assert len(border) == 16
    assert (counts <= 2).all()


def test_arch_index_and_class():
    # 目标 46：同侧 47..41 → 1..7，对侧 31..37 → 8..14
    assert [gen.arch_index(f, 46) for f in (47, 46, 45, 41, 31, 37)] == [1, 2, 3, 7, 8, 14]
    assert gen.target_class(11) == 0 and gen.target_class(27) == 6 and gen.target_class(36) == 13


def test_make_sample_centers_between_neighbors_and_mirrors_left_side():
    rng = np.random.default_rng(0)
    fdi = np.array([44, 45, 46, 47, 34, 35, 36, 37])
    centers = {44: (-14, 5), 45: (-16, 12), 46: (-19, 20), 47: (-22, 30), 34: (14, 5), 35: (16, 12), 36: (19, 20), 37: (22, 30)}
    pts = np.stack([rng.normal(0, 0.5, (gen.TOOTH_POINTS, 3)) + [*centers[int(f)], 0.0] for f in fdi]).astype(np.float32)
    rim = pts[:, : gen.RIM_POINTS]
    right = gen.make_sample(fdi, pts, rim, 2, np.random.default_rng(1), drop=False, augment=False)  # 46
    left = gen.make_sample(fdi, pts, rim, 6, np.random.default_rng(1), drop=False, augment=False)  # 36
    for ctx, ctx_idx, cls, gt, r in (right, left):
        assert ctx.shape == (gen.N_CTX, 3) and gt.shape == (gen.TOOTH_POINTS, 3) and r.shape == (gen.RIM_POINTS, 3)
        assert cls == gen.target_class(46)  # 左右同名牙是同一类
        assert np.linalg.norm(gt.mean(axis=0)) < 1.5  # 原点在两侧邻牙（45、47）质心的中点附近
    # 36 镜像后与 46 落在同一侧：真牙相对原点的偏移一致（数据本身左右对称）
    assert np.allclose(right[3].mean(axis=0), left[3].mean(axis=0), atol=0.2)


def test_model_forward_shapes():
    torch = pytest.importorskip("torch")
    model = gen.build_model(template="dome").eval()
    ctx = torch.randn(2, gen.N_CTX, 3) * 10
    idx = torch.randint(0, 16, (2, gen.N_CTX))
    with torch.no_grad():
        pred = model(ctx, idx, torch.tensor([2, 13]))
    assert pred.shape == (2, model.template.shape[0], 3)
    assert gen.chamfer(pred, torch.randn(2, 500, 3)).shape == (2,)
    assert model.laplacian(pred).shape[1] == int(model.interior.sum())


def test_pick_missing_keeps_enough_teeth_and_skips_third_molars():
    from dental_ai.eval_missing_seg import pick_missing

    labels = np.repeat([0, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47], 10)
    for seed in range(50):
        teeth = pick_missing(labels, np.random.default_rng(seed))
        assert 1 <= len(teeth) <= 3 and len(set(teeth)) == len(teeth)
        assert all(t % 10 != 8 and t in labels for t in teeth)
    assert len(pick_missing(labels, np.random.default_rng(0), n=2)) == 2
    assert pick_missing(np.repeat([0, 31, 32, 33, 41, 42, 43], 10), np.random.default_rng(0)) == ()  # 牙太少就不再去掉


def test_attention_segmentation_model_forward():
    torch = pytest.importorskip("torch")
    from dental_ai.segmentation.deep import build_model

    model = build_model(arch="dgcnn_attn").eval()
    with torch.no_grad():
        out = model(torch.randn(1, 15, 700))
    assert out.shape == (1, 17, 700)


def test_zero_contour_on_sphere_is_the_equator():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    loop = zero_contour(mesh, mesh.vertices[:, 2] - 1.0, smooth_iters=0)
    assert np.abs(loop[:, 2] - 1.0).max() < 0.05  # 线性插值的零点在 z = 1 上
    assert np.abs(np.linalg.norm(loop[:, :2], axis=1) - np.sqrt(24)).max() < 0.1
