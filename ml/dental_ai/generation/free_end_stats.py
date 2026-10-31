"""游离端（牙弓末端）缺失牙的位置和高度统计：从带标注的训练数据里学“末端那颗牙相对它近中邻牙”的偏移。

两侧最后一颗磨牙都缺时，没有对侧牙可以镜像，以前只能沿牙弓外推（位置偏 1.3–1.8 mm、比真牙低 2–3 mm）。
这里统计：在近中邻牙的局部坐标系（近中、颊侧、咬合方向）里，目标牙质心的偏移（按邻牙近远中宽度归一化）和
牙冠顶的高度差（mm）。计算邻牙坐标系时先把目标牙当作缺失，与推理时一致。

    python -m dental_ai.generation.free_end_stats --zips ../data/drive/raw/Teeth3DS --split <训练集划分> --out dental_ai/data/free_end_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..fdi import QUADRANTS, position

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "free_end_stats.json"


def _measure(args):
    from ..geometry import estimate_arch_frame
    from ..templates.frame import tooth_frame

    scan = args
    try:
        mesh, labels = scan.load()
        arch = estimate_arch_frame(mesh, labels, scan.jaw)
    except Exception:
        return []
    out = []
    present = set(np.unique(labels[labels > 0]).tolist())
    for q in QUADRANTS[scan.jaw]:
        for p in (6, 7, 8):  # 目标牙位；近中邻牙是 p - 1
            target, neighbor = q * 10 + p, q * 10 + p - 1
            if target not in present or neighbor not in present or q * 10 + p + 1 in present:
                continue  # 只统计真正在末端的牙
            missing = np.where(labels == target, 0, labels)
            try:
                frame = tooth_frame(mesh, missing, scan.jaw, neighbor, arch)
            except (ValueError, IndexError):
                continue
            v_n, v_t = mesh.vertices[labels == neighbor], mesh.vertices[labels == target]
            width = float(np.ptp(v_n @ frame.mesial))
            d = v_t.mean(axis=0) - v_n.mean(axis=0)
            offset = [float(d @ frame.mesial) / width, float(d @ frame.buccal) / width, float(d @ frame.occlusal) / width]
            dtop = float((v_t @ frame.occlusal).max() - (v_n @ frame.occlusal).max())
            out.append((f"{scan.jaw}_{p - 1}_{p}", offset, dtop))
    return out


def compute(scans, workers: int = 4) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [r for part in pool.map(_measure, scans, chunksize=2) for r in part]
    stats = {}
    for key in sorted({k for k, _, _ in rows}):
        offs = np.array([o for k, o, _ in rows if k == key])
        dtops = np.array([t for k, _, t in rows if k == key])
        stats[key] = {"n": len(offs), "offset_over_width": np.median(offs, axis=0).round(4).tolist(),
                      "dtop_mm": round(float(np.median(dtops)), 3)}
    return stats


def load(path: Path = DEFAULT_PATH) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    from ..datasets import teeth3ds_zip_scans

    p = argparse.ArgumentParser(description="统计游离端缺失牙相对近中邻牙的位置和高度")
    p.add_argument("--zips", type=Path, required=True)
    p.add_argument("--split", type=Path, nargs="+", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=DEFAULT_PATH)
    args = p.parse_args()
    scans = [s for f in args.split for s in teeth3ds_zip_scans(args.zips, f)]
    stats = compute(scans, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
