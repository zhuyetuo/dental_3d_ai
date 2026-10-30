import numpy as np
import pytest

from dental_ai.margin.cervical import cervical_line
from dental_ai.margin.detect import detect_margin, resample_loop
from dental_ai.pipeline import run
from dental_ai.segmentation.base import Segmenter
from dental_ai.synthetic import synthetic_arch


def radial_error(points, truth):
    cx, cy, r = truth
    return np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r)


@pytest.mark.parametrize("fdi", [46, 33, 31])
@pytest.mark.parametrize("offset", [0.6, -0.6])
def test_snaps_to_shoulder_edge(fdi, offset):
    """分割边界向外或向内偏 0.6 mm，检测结果应贴回肩台外缘。"""
    mesh, labels = synthetic_arch("lower", prepared=(fdi,), resolution=0.2, band=12, label_offset=offset)
    truth = mesh.metadata["margins"][fdi]
    initial = radial_error(cervical_line(mesh, labels, fdi), truth).mean()
    result = detect_margin(mesh, labels, fdi)
    err = radial_error(result.points, truth)
    assert initial > 0.5
    assert err.mean() < 0.2
    assert np.percentile(err, 90) < 0.35
    assert result.confidence > 0.8 and not result.warnings
    assert result.points.shape == (120, 3)


def test_resample_loop_is_uniform():
    t = np.linspace(0, 2 * np.pi, 37)[:-1]
    loop = np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=1)
    pts = resample_loop(loop, 90)
    seg = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    assert len(pts) == 90 and seg.std() / seg.mean() < 0.05


def test_pipeline_margins(tmp_path):
    mesh, labels = synthetic_arch("lower", missing=(36,), prepared=(46,), resolution=0.25, band=12, label_offset=0.5)

    class Fixed(Segmenter):
        name = "fixed"
        postprocess = False

        def predict(self, m, jaw):
            return labels

    report = run(mesh, "lower", Fixed(), tmp_path, prepared_teeth=[46, 36])
    assert report["margins"]["46"]["source"] == "auto"
    assert any("36" in w for w in report["warnings"])  # 36 不存在
    manual = {46: {"points": [[0, 0, 0]] * 4, "control": None}}
    report = run(mesh, "lower", Fixed(), tmp_path / "b", prepared_teeth=[46], margin_overrides=manual)
    assert report["margins"]["46"]["source"] == "manual"
    # 退化的颈缘线：牙冠生成失败，只记警告，不影响其它结果
    assert any("预备体牙冠生成失败" in w for w in report["warnings"])
