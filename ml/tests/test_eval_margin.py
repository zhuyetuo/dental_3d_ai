import numpy as np
import trimesh

from dental_ai.eval_margin import PREP, line_errors, shifted_labels, truth_line
from dental_ai.margin.cervical import cervical_line


def _open_tube(radius=3.0, step=0.1, sections=96):
    """圆管侧面，z 从 -3 到 3，每 step mm 一圈顶点；顶端封口（像代型的切端），底端开口。"""
    zs = np.arange(-3.0, 3.0 + 1e-9, step)
    a = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    verts = np.array([[radius * np.cos(t), radius * np.sin(t), z] for z in zs for t in a])
    faces = []
    for i in range(len(zs) - 1):
        for j in range(sections):
            p, q = i * sections + j, i * sections + (j + 1) % sections
            faces += [[p, q, p + sections], [q, q + sections, p + sections]]
    top = len(verts)
    verts = np.vstack([verts, [[0, 0, zs[-1]]]])
    last = (len(zs) - 1) * sections
    faces += [[last + j, last + (j + 1) % sections, top] for j in range(sections)]
    tube = trimesh.Trimesh(verts, faces, process=False)
    labels = (tube.triangles_center[:, 2] > 0).astype(int)  # z > 0 是“预备体”
    return tube, labels


def test_truth_line_and_shift():
    mesh, face_label = _open_tube()
    line = truth_line(mesh, face_label)
    assert np.abs(line[:, 2]).max() < 1e-6  # 交界就是 z = 0
    up = np.array([0.0, 0.0, 1.0])
    base = cervical_line(mesh, shifted_labels(mesh, face_label, line, up, 0.0), PREP)[:, 2].mean()
    moved = cervical_line(mesh, shifted_labels(mesh, face_label, line, up, 1.5), PREP)[:, 2].mean()
    assert 1.3 < moved - base < 1.7  # 初始线往预备体里收了约 1.5 mm

    from scipy.spatial import cKDTree

    err = line_errors(line[::5], cKDTree(line))
    assert err["mean"] < 1e-6 and err["within_0.2"] == 1.0
