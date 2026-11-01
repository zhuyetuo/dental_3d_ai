import json

import numpy as np

from dental_ai.external import fair_split, import_predictions, make_layout, to_fdi


def test_to_fdi_schemes():
    # TeethSplineSeg：1 = 右侧最后一颗磨牙，8 / 9 = 两颗中切牙
    assert to_fdi([0, 1, 8, 9, 16], "upper", "splineseg").tolist() == [0, 18, 11, 21, 28]
    assert to_fdi([1, 9], "lower", "splineseg").tolist() == [38, 41]
    # CrossTooth：1–8 左侧，9–16 右侧，17 / 18 是边界类 → 牙龈
    assert to_fdi([1, 8, 9, 16, 17, 18], "upper", "crosstooth").tolist() == [21, 28, 11, 18, 0, 0]
    assert to_fdi([1, 9], "lower", "crosstooth").tolist() == [31, 41]


def test_fair_split(tmp_path):
    t3 = tmp_path / "Teeth3DS_train_test_split"
    ch = tmp_path / "3DTeethSeg22_challenge_train_test_split"
    t3.mkdir()
    ch.mkdir()
    (t3 / "testing_lower.txt").write_text("A_lower\nB_lower\n")
    (t3 / "testing_upper.txt").write_text("C_upper\n")
    (t3 / "training_lower.txt").write_text("D_lower\n")
    (t3 / "training_upper.txt").write_text("")
    (ch / "private-testing-set.txt").write_text("A_lower\nC_upper\nD_lower\n")
    (ch / "public-training-set-1.txt").write_text("B_lower\n")
    assert fair_split(tmp_path) == ["A_lower", "C_upper"]


def test_layout_and_import(tmp_path):
    data = tmp_path / "data" / "X1"
    data.mkdir(parents=True)
    (data / "X1_lower.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 5 5 5\nf 1 2 3\n")  # 第 4 个是孤立顶点
    (data / "X1_lower.json").write_text(json.dumps({"jaw": "lower", "labels": [0, 31, 31, 0]}))
    root = tmp_path / "root"
    assert make_layout(tmp_path / "data", root) == ["X1_lower"]
    assert (root / "lower" / "X1" / "X1_lower.obj").exists()
    assert (root / "testing_lower.txt").read_text().split() == ["X1_lower"]
    assert (root / "training_upper.txt").read_text() == ""

    pred = tmp_path / "pred"
    pred.mkdir()
    np.savetxt(pred / "X1_lower.txt", [0, 8, 9, 0], fmt="%d")
    done, problems = import_predictions(pred, tmp_path / "data", tmp_path / "json", "splineseg")
    assert done == 1 and not problems
    assert json.loads((tmp_path / "json" / "X1_lower.json").read_text())["labels"] == [0, 31, 41, 0]

    np.savetxt(pred / "X1_lower.txt", [0, 8, 9], fmt="%d")  # 顶点数对不上要报出来
    done, problems = import_predictions(pred, tmp_path / "data", tmp_path / "json2", "splineseg")
    assert done == 0 and "OBJ 有 4 个" in problems[0]
