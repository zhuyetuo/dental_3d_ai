import json
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree

from dental_ai.evaluation import check_orientation, evaluate, find_scans, segmentation_metrics
from dental_ai.io import load_labels, save_labels
from dental_ai.synthetic import synthetic_arch


def test_perfect_prediction():
    mesh, labels = synthetic_arch("lower", missing=(36,))
    m, per_tooth = segmentation_metrics(labels, labels, mesh.vertices)
    for k, v in m.items():
        assert v == 1.0, k
    assert set(per_tooth) == set(np.unique(labels)) - {0}


def test_wrong_ids_and_gum_leak():
    mesh, labels = synthetic_arch("lower")
    pred = labels.copy()
    pred[labels == 31], pred[labels == 32] = 32, 31  # 编号互换：分割正确、识别错误
    m, _ = segmentation_metrics(labels, pred, mesh.vertices)
    assert m["tsa"] == 1.0 and m["inst_iou"] == 1.0
    assert m["tir"] == m["inst_tir"] == 12 / 14
    assert m["all_correct"] == 0.0
    assert m["t_miou"] < 1.0

    # 41 周围 1 mm 内的牙龈被分进 41（边界外溢）：分割变差，但识别不受影响
    pred = labels.copy()
    d, _ = cKDTree(mesh.vertices[labels == 41]).query(mesh.vertices)
    pred[(labels == 0) & (d < 1.0)] = 41
    m, _ = segmentation_metrics(labels, pred, mesh.vertices)
    assert m["tsa"] < 1.0 and m["tir"] == 1.0 and m["all_correct"] == 1.0


def _make_dataset(tmp_path):
    data, gt = tmp_path / "obj", tmp_path / "gt"
    for case, missing in (("CASE0001", ()), ("CASE0002", (36,))):
        for jaw in ("lower",):
            mesh, labels = synthetic_arch(jaw, missing=missing, band=12)
            (data / case).mkdir(parents=True, exist_ok=True)
            (gt / case).mkdir(parents=True, exist_ok=True)
            mesh.export(data / case / f"{case}_{jaw}.obj")
            save_labels(gt / case / f"{case}_{jaw}.json", labels, jaw)
    return data, gt


def test_evaluate_dataset_and_cli(tmp_path):
    data, gt = _make_dataset(tmp_path)
    split = tmp_path / "split.txt"
    split.write_text("CASE0002\n")
    assert [s.name for s in find_scans(data, gt)] == ["CASE0001_lower", "CASE0002_lower"]
    assert [s.name for s in find_scans(data, gt, split)] == ["CASE0002_lower"]

    summary = evaluate(find_scans(data, gt), lambda scan, mesh: load_labels(scan.gt_path)[0], tmp_path / "res")
    assert summary["n_scans"] == 2 and summary["all"]["t_miou"] == 1.0
    assert (tmp_path / "res" / "per_scan.csv").exists()

    # 命令行：预测目录用 TGNet 的平铺格式
    pred = tmp_path / "pred"
    pred.mkdir()
    for s in find_scans(data, gt):
        labels, _ = load_labels(s.gt_path)
        labels[labels == 31] = 32
        save_labels(pred / f"{s.name}.json", labels, "lower")
    out = tmp_path / "cli"
    subprocess.run(
        [sys.executable, "-m", "dental_ai.evaluation", "--data-dir", data, "--gt-dir", gt, "--pred-dir", pred, "--out", out],
        check=True,
        capture_output=True,
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["all"]["all_correct"] == 0.0 and summary["per_fdi_iou"]["31"] == 0.0


def test_missing_prediction_is_recorded(tmp_path):
    data, gt = _make_dataset(tmp_path)

    def predict(scan, mesh):
        if scan.case == "CASE0001":
            raise FileNotFoundError("没有预测结果")
        return load_labels(scan.gt_path)[0]

    summary = evaluate(find_scans(data, gt), predict, tmp_path / "res")
    assert summary["n_scans"] == 1 and summary["failures"][0]["scan"] == "CASE0001_lower"


def test_check_orientation(tmp_path):
    data, gt = _make_dataset(tmp_path)
    summary = check_orientation(find_scans(data, gt), tmp_path / "res", trials=2)
    assert summary["convention_ok_rate"] == 1.0
    assert summary["within_10deg_rate"] == 1.0
