import numpy as np

from dental_ai.io import load_mesh


def test_obj_keeps_isolated_vertices_and_order(tmp_path):
    # 真实 Teeth3DS 扫描里有没被面引用的孤立顶点，标签按全部 v 行编号，读入时不能丢
    path = tmp_path / "scan_lower.obj"
    path.write_text(
        "# test\nv 0 0 0 0.5 0.5 0.5\nv 9 9 9\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
        "vn 0 0 1\nf 1//1 3//1 4//1\nf 3 5 4\nf -3 -1 -2\n"  # 负索引
    )
    m = load_mesh(path)
    assert len(m.vertices) == 5 and np.allclose(m.vertices[1], 9)
    assert m.faces.tolist() == [[0, 2, 3], [2, 4, 3], [2, 4, 3]]


def test_obj_polygons_are_triangulated(tmp_path):
    path = tmp_path / "quad.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1/1/1 2/2/1 3/3/1 4/4/1\n")
    assert load_mesh(path).faces.tolist() == [[0, 1, 2], [0, 2, 3]]


def test_boundary_loops_split_at_pinch_vertex():
    # 两个三角形只共享一个顶点（真实口扫的边界上常见）：应得到两个完整的环，而不是只找到一个
    import trimesh

    from dental_ai.margin.cervical import boundary_loops

    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0.0]])
    m = trimesh.Trimesh(v, [[0, 1, 2], [0, 3, 4]], process=False)
    loops = boundary_loops(m)
    assert sorted(len(x) for x in loops) == [3, 3]


def test_clean_crown_fills_small_holes_and_drops_islands():
    import trimesh

    from dental_ai.geometry import clean_crown
    from dental_ai.margin.cervical import boundary_loops

    grid = trimesh.creation.box(extents=[10, 10, 2])
    grid = grid.subdivide().subdivide()
    keep = grid.face_normals[:, 2] > -0.5  # 去掉底面 → 一个大开口
    top = np.nonzero(grid.face_normals[:, 2] > 0.5)[0]
    keep[top[np.argmin(np.linalg.norm(grid.triangles_center[top, :2], axis=1))]] = False  # 顶面挖掉一个三角形
    crown = trimesh.Trimesh(grid.vertices, grid.faces[keep], process=False)
    island = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    island.apply_translation([20, 0, 0])
    crown = trimesh.util.concatenate([crown, island])
    assert len(boundary_loops(crown)) >= 2
    cleaned = clean_crown(crown)
    assert len(boundary_loops(cleaned)) == 1 and cleaned.is_winding_consistent
    assert cleaned.vertices[:, 0].max() < 10  # 小碎片去掉了
