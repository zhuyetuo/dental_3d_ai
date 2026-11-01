"""接入外部分割模型（TeethSplineSeg、CrossTooth 等）做对比评估的数据转换工具。

外部模型各有自己的输入目录结构和输出格式，这里负责：
1. fair-split：算出“所有模型都没见过”的测试病例。Teeth3DS 官方划分（Teeth3DS_train_test_split，TeethSplineSeg 用）
   和 3DTeethSeg'22 挑战赛划分（TGNet 用）大量交叉：各自 600 个测试扫描里有 426 个在对方的训练集里，
   两个测试集的交集只有 174 个扫描，只有它们对两边都是没见过的；
2. layout：把我们导出的数据（<病例>/<病例>_<颌>.obj/json）链接成 Teeth3DS 原始结构（<颌>/<病例>/...）+ 划分文件；
3. import：把外部模型逐顶点的类别号（txt，一行一个）转成我们评估用的 FDI 标注 JSON（evaluation --pred-dir）。

    python -m dental_ai.external fair-split --splits ../data/drive/raw/Teeth3DS/Teeth3DS+ --out ../data/fair_test.txt
    python -m dental_ai.external layout --data-dir ../data/fair_test --out ../data/splineseg_root
    python -m dental_ai.external import --scheme splineseg --pred-dir <txt 目录> --data-dir ../data/fair_test --out <json 目录>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# TeethSplineSeg（evaluate.py / postprocess_fast.py）：上下颌共用 1–16，从患者右侧最后一颗磨牙到左侧最后一颗磨牙
SPLINESEG_UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
SPLINESEG_LOWER = [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]
# CrossTooth（utils.py 的 label2color_*）：1–8 = 左侧 1–8 号牙（UL / LL），9–16 = 右侧（UR / LR）；17、18 是边界类，算牙龈
CROSSTOOTH_UPPER = [21, 22, 23, 24, 25, 26, 27, 28, 11, 12, 13, 14, 15, 16, 17, 18]
CROSSTOOTH_LOWER = [31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48]
SCHEMES = {
    "splineseg": {"upper": SPLINESEG_UPPER, "lower": SPLINESEG_LOWER},
    "crosstooth": {"upper": CROSSTOOTH_UPPER, "lower": CROSSTOOTH_LOWER},
}


def _read_list(*paths: Path) -> list[str]:
    out = []
    for p in paths:
        out += [line.strip() for line in p.read_text().splitlines() if line.strip()]
    return out


def fair_split(splits_root: Path) -> list[str]:
    """两个官方测试集的交集（“病例_颌位”），也就是两套训练集都没用过的扫描。"""
    t3 = splits_root / "Teeth3DS_train_test_split"
    ch = splits_root / "3DTeethSeg22_challenge_train_test_split"
    t3_test = set(_read_list(t3 / "testing_lower.txt", t3 / "testing_upper.txt"))
    t3_train = set(_read_list(t3 / "training_lower.txt", t3 / "training_upper.txt"))
    ch_test = set(_read_list(ch / "private-testing-set.txt"))
    ch_train = set(_read_list(*sorted(ch.glob("public-training-set-*.txt"))))
    fair = (t3_test & ch_test) - t3_train - ch_train
    return sorted(fair)


def make_layout(data_dir: Path, out_root: Path) -> list[str]:
    """data_dir 里的 <病例>/<病例>_<颌>.obj/json → out_root/<颌>/<病例>/<病例>_<颌>.obj/json（软链接），
    并写 testing_*.txt（全部扫描）和空的 training_*.txt。返回扫描名列表。"""
    names = {"lower": [], "upper": []}
    for obj in sorted(data_dir.rglob("*.obj")):
        case, jaw = obj.stem.rsplit("_", 1)
        if jaw not in names:
            continue
        gt = obj.with_suffix(".json")
        dest = out_root / jaw / case
        dest.mkdir(parents=True, exist_ok=True)
        for src in (obj, gt):
            link = dest / src.name
            if src.exists() and not link.exists():
                os.symlink(src.resolve(), link)
        names[jaw].append(obj.stem)
    for jaw, items in names.items():
        (out_root / f"testing_{jaw}.txt").write_text("\n".join(items) + ("\n" if items else ""))
        (out_root / f"training_{jaw}.txt").write_text("")
    return names["lower"] + names["upper"]


def to_fdi(classes: np.ndarray, jaw: str, scheme: str) -> np.ndarray:
    table = np.array([0] + SCHEMES[scheme][jaw])
    classes = np.asarray(classes, dtype=np.int64)
    valid = (classes >= 1) & (classes <= 16)
    return np.where(valid, table[np.clip(classes, 0, 16)], 0)


def _count_obj_vertices(path: Path) -> int:
    with open(path, errors="replace") as f:
        return sum(1 for line in f if line.startswith("v "))


def import_predictions(pred_dir: Path, data_dir: Path, out_dir: Path, scheme: str) -> tuple[int, list[str]]:
    """pred_dir/<病例>_<颌>.txt（逐顶点类别号，与 OBJ 的 v 行一一对应）→ out_dir/<病例>_<颌>.json。
    返回（转换数，缺失或顶点数对不上的扫描）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    done, problems = 0, []
    for obj in sorted(data_dir.rglob("*.obj")):
        stem = obj.stem
        jaw = stem.rsplit("_", 1)[-1]
        txt = pred_dir / f"{stem}.txt"
        if jaw not in ("upper", "lower"):
            continue
        if not txt.exists():
            problems.append(f"{stem}: 没有预测")
            continue
        classes = np.loadtxt(txt, dtype=np.int64).reshape(-1)
        n = _count_obj_vertices(obj)
        if len(classes) != n:
            problems.append(f"{stem}: 预测 {len(classes)} 个顶点，OBJ 有 {n} 个")
            continue
        labels = to_fdi(classes, jaw, scheme).tolist()
        (out_dir / f"{stem}.json").write_text(json.dumps({"jaw": jaw, "labels": labels, "instances": labels}))
        done += 1
    return done, problems


def main() -> None:
    p = argparse.ArgumentParser(description="外部分割模型对比评估的数据转换")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fair-split", help="两套官方测试集的交集（所有模型都没见过的扫描）")
    a.add_argument("--splits", type=Path, required=True, help="Teeth3DS+ 目录（含两套划分文件夹）")
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("layout", help="链接成 Teeth3DS 原始目录结构 + 划分文件")
    b.add_argument("--data-dir", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("import", help="外部模型逐顶点类别号 → FDI 标注 JSON")
    c.add_argument("--scheme", choices=sorted(SCHEMES), required=True)
    c.add_argument("--pred-dir", type=Path, required=True)
    c.add_argument("--data-dir", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "fair-split":
        fair = fair_split(args.splits)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(fair) + "\n")
        print(f"{len(fair)} 个扫描（两套官方训练集都没用过）→ {args.out}")
    elif args.cmd == "layout":
        names = make_layout(args.data_dir, args.out)
        print(f"{len(names)} 个扫描 → {args.out}")
    else:
        done, problems = import_predictions(args.pred_dir, args.data_dir, args.out, args.scheme)
        print(f"转换 {done} 个；问题 {len(problems)} 个")
        for line in problems[:20]:
            print("  " + line)
        if not done:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
