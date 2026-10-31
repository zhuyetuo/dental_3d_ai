from dataclasses import dataclass

import numpy as np

from dental_ai.segmentation.baseline import BaselineSegmenter, class_to_fdi, fdi_to_class, train
from dental_ai.synthetic import synthetic_arch


@dataclass
class SyntheticScan:
    name: str
    jaw: str
    missing: tuple

    def load(self):
        return synthetic_arch(self.jaw, missing=self.missing, band=12)


def test_class_mapping_round_trip():
    for jaw in ("lower", "upper"):
        fdi = np.array([0, 11, 18, 21, 28, 31, 38, 41, 48])
        cls = fdi_to_class(fdi, jaw)
        same_jaw = np.isin(fdi // 10, [1, 2] if jaw == "upper" else [3, 4])
        assert (class_to_fdi(cls, jaw)[same_jaw] == fdi[same_jaw]).all()
        assert (cls[~same_jaw] == 0).all()  # 另一颌的牙号和牙龈都归到 0


def test_train_and_predict_on_synthetic(tmp_path):
    scans = [SyntheticScan(f"s{i}", jaw, missing) for i, (jaw, missing) in enumerate(
        [("lower", ()), ("lower", (36,)), ("upper", ()), ("lower", (45,))])]
    info = train(scans, tmp_path / "m.joblib", per_scan=2000, workers=1, max_iter=25, early_stopping=False)
    assert info["rows"] == 8000
    mesh, gt = synthetic_arch("lower", missing=(46,), band=12)
    pred = BaselineSegmenter(tmp_path / "m.joblib").predict(mesh, "lower")
    teeth = gt > 0
    assert ((pred > 0) == teeth).mean() > 0.9  # 牙齿 / 牙龈
    assert (pred[teeth] == gt[teeth]).mean() > 0.6  # 牙号（合成数据很少，只验证链路通）
