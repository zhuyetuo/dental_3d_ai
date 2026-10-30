import pytest

from dental_ai.segmentation.remote import RemoteSegmenter
from dental_ai.synthetic import synthetic_arch


def test_remote_segmenter_round_trip(fake_segmenter_url):
    mesh, labels = synthetic_arch("lower", missing=(36,))
    pred = RemoteSegmenter(fake_segmenter_url).predict(mesh, "lower")
    assert (pred == labels).all()


def test_remote_segmenter_errors(fake_segmenter_url):
    mesh, _ = synthetic_arch("lower")
    with pytest.raises(RuntimeError, match="400"):
        RemoteSegmenter(fake_segmenter_url).predict(mesh, "upper")
    with pytest.raises(RuntimeError, match="无法连接"):
        RemoteSegmenter("http://127.0.0.1:9", timeout=2).predict(mesh, "lower")
