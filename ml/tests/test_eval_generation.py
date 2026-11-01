import numpy as np
import trimesh

from dental_ai.eval_generation import compare


def test_compare_identical_and_shifted():
    real = trimesh.creation.icosphere(subdivisions=3, radius=4.0)
    up = np.array([0.0, 0.0, 1.0])
    same = compare(real, real, up)
    assert same["chamfer_l1"] < 0.15
    assert same["fscore_0.5"] > 0.95

    shifted = real.copy()
    shifted.apply_translation([1.0, 0, 0])
    far = compare(shifted, real, up)
    assert far["chamfer_l1"] > 0.4
    assert far["fscore_0.3"] < same["fscore_0.3"]
    assert abs(far["position"] - 1.0) < 0.2
