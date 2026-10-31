"""深度分割模型的链路测试（没有安装 PyTorch 时跳过，CI 里就是跳过的）。"""

from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dental_ai.segmentation.baseline import N_CLASSES  # noqa: E402
from dental_ai.segmentation.deep import DeepSegmenter, build_cache, build_model, train  # noqa: E402
from dental_ai.synthetic import synthetic_arch  # noqa: E402


@dataclass
class SyntheticScan:
    name: str
    jaw: str
    missing: tuple

    def load(self):
        return synthetic_arch(self.jaw, missing=self.missing, band=12)


def test_model_shapes():
    model = build_model(in_dim=15, k=8)
    out = model(torch.randn(2, 15, 256))
    assert out.shape == (2, N_CLASSES, 256)


def test_cache_train_predict(tmp_path):
    scans = [SyntheticScan(f"s{i}_lower", "lower", m) for i, m in enumerate([(), (36,), (45,), ()])]
    assert build_cache(scans, tmp_path / "cache", workers=1) == 4
    info = train(tmp_path / "cache", tmp_path / "m.pt", epochs=1, n_points=512, batch_size=2, device="cpu")
    assert info["train_scans"] == 3
    mesh, _ = synthetic_arch("lower", band=12)
    labels = DeepSegmenter(tmp_path / "m.pt", device="cpu", passes=1, refine_iters=2).predict(mesh, "lower")
    assert labels.shape == (len(mesh.vertices),)
    assert set(np.unique(labels)) <= {0} | set(range(31, 39)) | set(range(41, 49))
