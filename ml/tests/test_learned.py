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
    for ctx, ctx_idx, cls, gt, r, ant, has_ant, nb, has_nb in (right, left):
        assert nb.shape == (2, gen.NB_POINTS, 3) and has_nb.tolist() == [1, 1]  # 45、47 都在：两侧邻牙都有
        assert has_ant == 0 and ant.shape == (gen.N_ANT, 6)  # 没给对颌
        assert ctx.shape == (gen.N_CTX, 3) and gt.shape == (gen.TOOTH_POINTS, 3) and r.shape == (gen.RIM_POINTS, 3)
        assert cls == gen.target_class(46)  # 左右同名牙是同一类
        assert np.linalg.norm(gt.mean(axis=0)) < 1.5  # 原点在两侧邻牙（45、47）质心的中点附近
    # 36 镜像后与 46 落在同一侧：真牙相对原点的偏移一致（数据本身左右对称）
    assert np.allclose(right[3].mean(axis=0), left[3].mean(axis=0), atol=0.2)


def test_make_sample_crops_antagonist_around_gap_and_mirrors_it():
    rng = np.random.default_rng(0)
    fdi = np.array([44, 45, 46, 47, 34, 35, 36, 37])
    centers = {44: (-14, 5), 45: (-16, 12), 46: (-19, 20), 47: (-22, 30), 34: (14, 5), 35: (16, 12), 36: (19, 20), 37: (22, 30)}
    pts = np.stack([rng.normal(0, 0.5, (gen.TOOTH_POINTS, 3)) + [*centers[int(f)], 0.0] for f in fdi]).astype(np.float32)
    rim = pts[:, : gen.RIM_POINTS]
    # 对颌：z = 2 的一个平面，法向朝上（绕序反了的口扫）→ antagonist_points 统一成朝下
    xy = rng.uniform(-40, 40, (60000, 2))
    ant = gen.antagonist_points(np.column_stack([xy, np.full(len(xy), 2.0)]), np.tile([0.0, 0.0, 1.0], (len(xy), 1)))
    assert np.allclose(ant[:, 5], -1)
    right = gen.make_sample(fdi, pts, rim, 2, np.random.default_rng(1), drop=False, augment=False, ant=ant)
    left = gen.make_sample(fdi, pts, rim, 6, np.random.default_rng(1), drop=False, augment=False, ant=ant)
    for sample in (right, left):
        a, has_ant = sample[5], sample[6]
        assert has_ant == 1 and a.shape == (gen.N_ANT, 6)
        assert np.linalg.norm(a[:, :2], axis=1).max() < gen.ANT_RADIUS  # 已减去原点，只留缺隙上方
        assert np.allclose(a[:, 2], 2.0, atol=0.3) and np.allclose(a[:, 5], -1)
    # 对颌离缺隙很远：当作没有
    far = ant.copy()
    far[:, 0] += 200
    assert gen.make_sample(fdi, pts, rim, 2, np.random.default_rng(1), drop=False, augment=False, ant=far)[6] == 0


def test_antagonist_model_forward_and_distance():
    torch = pytest.importorskip("torch")
    model = gen.build_model(template="dome", antagonist=True, refine=True).eval()
    ctx, idx, cls = torch.randn(2, gen.N_CTX, 3), torch.randint(0, 16, (2, gen.N_CTX)), torch.tensor([3, 12])
    ant = torch.cat([torch.randn(2, gen.N_ANT, 3), torch.nn.functional.normalize(torch.randn(2, gen.N_ANT, 3), dim=-1)], dim=-1)
    with torch.no_grad():
        with_ant = model(ctx, idx, cls, ant, torch.tensor([1.0, 0.0]))
        without = model(ctx, idx, cls)
    assert with_ant.shape == without.shape == (2, model.template.shape[0], 3)
    assert torch.allclose(with_ant[1], without[1], atol=1e-5)  # has_ant = 0 的样本和不给对颌一样
    # 平面 z = 1（法向朝下）下方 0.4 mm 的点：距离 +0.4；上方 0.2 mm：−0.2；离得远：inf
    xy = torch.rand(1, 4000, 2) * 10
    plane = torch.cat([xy, torch.ones(1, 4000, 1), torch.zeros(1, 4000, 2), -torch.ones(1, 4000, 1)], dim=-1)
    pts = torch.tensor([[[5.0, 5.0, 0.6], [5.0, 5.0, 1.2], [5.0, 5.0, -9.0]]])
    d = gen.antagonist_distance(pts, plane)[0]
    assert abs(float(d[0]) - 0.4) < 1e-4 and abs(float(d[1]) + 0.2) < 1e-4 and torch.isinf(d[2])


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


def test_prep_side_from_margin_point_cloud():
    import trimesh

    from dental_ai.margin.learned import prep_side

    mesh = trimesh.creation.icosphere(subdivisions=5, radius=5.0)
    a = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    r = np.sqrt(25 - 3.0 ** 2)
    line = np.c_[r * np.cos(a), r * np.sin(a), np.full_like(a, 3.0)]  # z = 3 的一圈：上面的小帽是预备体
    side = prep_side(mesh, line)
    z = mesh.vertices[:, 2]
    assert side[z > 3.4].all() and not side[z < 2.6].any()


def _open_cap(radius=5.0, bumps=0.0):
    import trimesh

    m = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    v = m.vertices * (1 + bumps * np.sin(6 * np.arctan2(m.vertices[:, 1], m.vertices[:, 0]))[:, None])
    keep = m.vertices[:, 2] > -1.0
    f = m.faces[keep[m.faces].all(axis=1)]
    remap = np.cumsum(keep) - 1
    return v[keep].astype(np.float32), remap[f].astype(np.int32)


def test_detail_library_fits_a_real_crown_onto_the_prediction(tmp_path):
    from dental_ai.generation.detail import DetailLibrary, _save_class, n_boundaries

    bumpy, plain_far = _open_cap(5.0, bumps=0.03), _open_cap(8.0)
    assert n_boundaries(bumpy[1]) == 1
    holed = bumpy[1][5:]  # 去掉几个三角形：多出一条开放边界
    assert n_boundaries(holed) > 1
    crowns = [(v - v.mean(axis=0), f) for v, f in (bumpy, plain_far)]
    _save_class(tmp_path, 5, crowns, [1, 0])
    pred, _ = _open_cap(5.2)
    pred = pred + [10.0, -3.0, 2.0]
    lib = DetailLibrary(tmp_path)
    verts, faces, info = lib.refine(pred, 5)
    assert info["detail_source"] == "fdi16" and len(faces) == len(bumpy[1])  # 选了大小接近的那颗
    assert info["detail_fit_mm"] < 0.25
    assert np.linalg.norm(verts.mean(axis=0) - pred.mean(axis=0)) < 0.3  # 落在预测的位置上
    assert lib.refine(pred, 6) is None  # 这一类没有库
    assert DetailLibrary(tmp_path, max_fit_mm=1e-6).refine(pred, 5) is None  # 贴不上就保留原来的牙冠


def test_real_missing_report_parser():
    from dental_ai.eval_real_missing import parse_report

    miss, present, _ = parse_report("There is maxillary constriction in the absence of crossbite. The molar class cannot be assessed due to the absence of tooth 46.")
    assert miss == {46} and not present  # “absence of crossbite” 不是缺牙
    miss, _, _ = parse_report("Sagittally it cannot be assessed due to the absence from the arch of teeth 16-15-26-36-37.")
    assert miss == {16, 15, 26, 36, 37}
    miss, _, _ = parse_report("Teeth 15, 26, 36 and 37 are absent from the arch; tooth 13 is in crossbite.")
    assert miss == {15, 26, 36, 37}
    miss, present, _ = parse_report("Tooth 45 is not present in the arch, with the corresponding deciduous 85 still present in the arch.")
    assert miss == {45} and present == {85}
    miss, present, _ = parse_report("The patient presents agenesis of teeth 35-45, with the deciduous teeth 75-85 still present in the arch.")
    assert miss == {35, 45} and present == {75, 85}
    assert parse_report("Overjet is increased and tooth 23 is in crossbite.")[0] == set()


def test_zero_contour_on_sphere_is_the_equator():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=5.0)
    loop = zero_contour(mesh, mesh.vertices[:, 2] - 1.0, smooth_iters=0)
    assert np.abs(loop[:, 2] - 1.0).max() < 0.05  # 线性插值的零点在 z = 1 上
    assert np.abs(np.linalg.norm(loop[:, :2], axis=1) - np.sqrt(24)).max() < 0.1


def test_simulate_closed_gap():
    """去掉 35 后 36 前移合拢，34 不动，牙号不变，不留 35。"""
    import numpy as np
    import trimesh
    from dental_ai.synthetic import pick_closed_gap, simulate_closed_gap

    xs, ys = np.arange(0, 40.01, 0.5), np.arange(0, 6.01, 0.5)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    v = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    n, m = len(xs), len(ys)
    idx = np.arange(n * m).reshape(n, m)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    faces = np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    labels = np.zeros(len(v), int)
    for fdi, lo in ((34, 5), (35, 13), (36, 21)):
        sel = (v[:, 0] >= lo) & (v[:, 0] < lo + 7) & (abs(v[:, 1] - 3) < 2)
        labels[sel] = fdi
        v[sel, 2] = 3.0
    mesh = trimesh.Trimesh(v, faces, process=False)
    m2, l2 = simulate_closed_gap(mesh, labels, "lower", (35,))
    assert set(l2.tolist()) == {0, 34, 36}
    assert (l2 == 36).sum() == (labels == 36).sum() and (l2 == 34).sum() == (labels == 34).sum()
    v2 = np.asarray(m2.vertices)
    assert abs(v2[l2 == 34, 0].mean() - v[labels == 34, 0].mean()) < 1e-6
    shift = v[labels == 36, 0].mean() - v2[l2 == 36, 0].mean()
    assert 5 < shift < 8
    assert len(m2.faces) < len(faces)

    full = np.array([q * 10 + p for q in (3, 4) for p in range(1, 8)])
    rng = np.random.default_rng(0)
    for _ in range(50):
        teeth = pick_closed_gap(full, rng)
        assert all(t in full and t % 10 < 7 for t in teeth) and len(teeth) <= 2


def test_merge_split_teeth():
    """一颗牙被分成两个牙号（一大一小、切口很长）要合并；真正相邻的两颗牙不合并。"""
    import numpy as np
    import trimesh
    from dental_ai.segmentation.postprocess import merge_split_teeth

    xs, ys = np.arange(0, 45.01, 0.5), np.arange(0, 6.01, 0.5)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    v = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    idx = np.arange(len(xs) * len(ys)).reshape(len(xs), len(ys))
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    mesh = trimesh.Trimesh(v, np.vstack([np.column_stack([a, b, c]), np.column_stack([a, c, d])]), process=False)
    block = lambda lo, hi: (v[:, 0] >= lo) & (v[:, 0] < hi) & (abs(v[:, 1] - 3) < 2)
    labels = np.zeros(len(v), int)
    labels[block(3, 10)] = 33
    labels[block(11, 18)] = 34
    labels[block(19, 30)] = 36
    labels[block(31, 41)] = 37
    out, merged = merge_split_teeth(mesh, labels, "lower", points=v)
    assert merged == [] and (out == labels).all()

    split = labels.copy()
    split[block(11, 18) & (v[:, 1] >= 4.0)] = 35  # 34 的舌侧一条被标成了 35
    out, merged = merge_split_teeth(mesh, split, "lower", points=v)
    assert merged == [(35, 34)] and (out == labels).all()

    touching = labels.copy()
    touching[block(10, 11)] = 33  # 33 和 34 直接挨着，但合起来有两颗牙宽
    out, merged = merge_split_teeth(mesh, touching, "lower", points=v)
    assert merged == []


def test_crown_quality():
    """评分卡：已知间隙的两个盒子、已知穿入深度的对颌平面、有沟的咬合面。"""
    import numpy as np
    import trimesh
    from dental_ai.crown_quality import occlusal_table, score_crown

    up = np.array([0.0, 0.0, 1.0])
    box = lambda cx, cz=0.0: trimesh.creation.box(extents=(8, 8, 6), transform=trimesh.transformations.translation_matrix((cx, 0, cz)))
    crown = box(0).subdivide().subdivide().subdivide()
    mesial, distal = box(-8.3), box(8.0 - 0.2)  # 一侧 0.3 mm 的缝，一侧挤进 0.2 mm
    plane = trimesh.creation.box(extents=(30, 30, 2), transform=trimesh.transformations.translation_matrix((0, 0, 3 + 1 - 0.4)))  # 对颌压进牙冠 0.4 mm
    row = score_crown(crown, up, mesial, distal, antagonist=plane.subdivide().subdivide().subdivide())
    assert abs(row["mesial_gap"] - 0.3) < 0.05 and abs(row["distal_gap"] + 0.2) < 0.05
    assert row["mesial_contact_area"] == 0 and row["distal_contact_area"] > 0
    assert abs(row["occl_clearance"] + 0.4) < 0.05 and row["occl_penetration_area"] > 30
    assert abs(row["height_vs_neighbors"]) < 0.05 and row["arch_offset"] < 0.05 and row["watertight"]

    lifted = score_crown(box(0, 1.0).subdivide().subdivide().subdivide(), up, mesial, distal)
    assert abs(lifted["height_vs_neighbors"] - 1.0) < 0.05

    rng = np.random.default_rng(0)
    xy = rng.uniform(-5, 5, (40000, 2))
    flat = np.column_stack([xy, np.zeros(len(xy))])
    groove = flat.copy()
    groove[:, 2] = -1.0 * np.exp(-(xy[:, 0] ** 2) / (2 * 0.4**2))  # 一条 1 mm 深的沟
    a, b = occlusal_table(flat, up), occlusal_table(groove, up)
    assert a["relief"] < 0.01 and a["detail"] < 0.01
    assert b["relief"] > 0.3 and b["detail"] > 0.1


def test_export_colored_ply_with_face_attributes(tmp_path):
    """STL 口扫自带逐面属性，面数变了以后导出不能报错。"""
    import numpy as np
    import trimesh
    from dental_ai.io import export_colored_ply

    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh.face_attributes["attr"] = np.zeros(len(mesh.faces) + 3, dtype=np.uint16)
    export_colored_ply(tmp_path / "a.ply", mesh, np.zeros(len(mesh.vertices), int))
    assert (tmp_path / "a.ply").stat().st_size > 0


def test_adjust_occlusion_lifts_or_reduces_to_target():
    """够不着对颌就抬上去，穿过去就磨掉，最后离对颌的最小距离等于目标；牙颈一圈不动。"""
    import numpy as np
    import trimesh
    from dental_ai.generation.occlusion import adjust_occlusion

    up = np.array([0.0, 0.0, 1.0])

    def cap():
        m = trimesh.creation.icosphere(subdivisions=4, radius=5.0)
        keep = (m.vertices[:, 2] > -1.0)[m.faces].all(axis=1)
        return m.submesh([np.flatnonzero(keep)], append=True)

    def plane(z):  # 对颌：下表面在高度 z 的一块板
        return trimesh.creation.box(extents=(30, 30, 2), transform=trimesh.transformations.translation_matrix((0, 0, z + 1))).subdivide().subdivide().subdivide().subdivide()

    for z, lifted in ((5.5, True), (4.6, False)):
        crown = cap()
        bottom = crown.vertices[:, 2] < -0.9
        before = crown.vertices.copy()
        info, warnings = adjust_occlusion(crown, plane(z), up, target=0.1)
        assert not warnings and abs(info["occlusal_clearance_mm"] - 0.1) < 0.03
        assert abs(z - crown.vertices[:, 2].max() - 0.1) < 0.03
        assert (info["occlusal_lift_mm"] > 0) == lifted and info["occlusal_reduction_mm"] > 0.1
        assert np.allclose(crown.vertices[bottom], before[bottom])
        assert ((z - 0.1 - crown.vertices[:, 2]) < 0.05).sum() > 20  # 不是一个点接触，是一小片

    far = cap()
    info, warnings = adjust_occlusion(far, plane(8.0), up, target=0.1)  # 差 3 mm：不抬
    assert info["occlusal_lift_mm"] == 0 and warnings
