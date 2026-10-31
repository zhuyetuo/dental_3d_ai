"""牙齿关键点（Teeth3DS+ / 3DTeethLand'24 的标注）。

文件 <病例>_<颌位>__kpt.json：{"key": "<病例>_<颌位>.obj", "objects": [{"class": ..., "coord": [x, y, z]}, ...]}，
坐标与扫描相同。类别（每颗牙）：
    Mesial / Distal          近中、远中接触点（各 1 个）
    Cusp                     牙尖（磨牙、前磨牙多个；切牙、尖牙没有或 1 个）
    OuterPoint / InnerPoint  颊（唇）侧、舌侧的牙颈附近点
    FacialPoint              唇颊面中点
标注里没有写点属于哪颗牙，按最近顶点的牙号分配（assign_to_teeth）。

    python -m dental_ai.keypoints --zips ../data/drive/raw/Teeth3DS --out ../eval_keypoints
统计：关键点到网格表面的距离、每颗牙各类点的个数、用 Mesial/Distal 测出的实际近远中宽度（与 fdi.expected_width
的教科书值比较），以及牙齿局部坐标系（templates/frame.py）的近中方向与标注的偏差。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

CLASSES = ("Mesial", "Distal", "Cusp", "OuterPoint", "InnerPoint", "FacialPoint")


def parse_keypoints(data: dict) -> list[tuple[str, np.ndarray]]:
    return [(o["class"], np.asarray(o["coord"], dtype=float)) for o in data.get("objects", [])]


def load_keypoints(path: str | Path) -> list[tuple[str, np.ndarray]]:
    return parse_keypoints(json.loads(Path(path).read_text()))


def assign_to_teeth(mesh: trimesh.Trimesh, labels: np.ndarray, keypoints, max_distance: float = 1.0) -> dict[int, dict[str, list[np.ndarray]]]:
    """{牙号: {类别: [点]}}。点按最近的牙齿顶点归类；离牙齿表面超过 max_distance 的点丢弃。"""
    teeth = np.nonzero(labels > 0)[0]
    tree = cKDTree(mesh.vertices[teeth])
    out: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for cls, p in keypoints:
        d, i = tree.query(p)
        if d <= max_distance:
            out[int(labels[teeth[i]])][cls].append(p)
    return out


def analyze(items) -> dict:
    """items：(扫描名, 颌位, 网格, 标签, 关键点 JSON) 的可迭代对象（datasets.iter_teeth3ds）。"""
    from .fdi import expected_width, position
    from .geometry import estimate_arch_frame
    from .templates.frame import tooth_frame

    surface, counts, widths, frame_err = [], defaultdict(lambda: defaultdict(list)), defaultdict(list), defaultdict(list)
    n_scans = 0
    for name, jaw, mesh, labels, kpt in items:
        if not kpt:
            continue
        n_scans += 1
        points = parse_keypoints(kpt)
        surface += list(trimesh.proximity.closest_point(mesh, np.array([p for _, p in points]))[1])
        per_tooth = assign_to_teeth(mesh, labels, points)
        try:
            arch = estimate_arch_frame(mesh, labels, jaw)
        except ValueError:
            arch = None
        for fdi, by_class in per_tooth.items():
            pos = position(fdi)
            for cls in CLASSES:
                counts[pos][cls].append(len(by_class.get(cls, [])))
            if len(by_class.get("Mesial", [])) == 1 and len(by_class.get("Distal", [])) == 1:
                m, d = by_class["Mesial"][0], by_class["Distal"][0]
                widths[(jaw, pos)].append(float(np.linalg.norm(m - d)))
                if arch is not None:
                    try:
                        frame = tooth_frame(mesh, labels, jaw, fdi, arch)
                    except (ValueError, IndexError):
                        continue
                    md = m - d
                    md -= (md @ frame.occlusal) * frame.occlusal
                    cos = md @ frame.mesial / max(np.linalg.norm(md), 1e-9)
                    frame_err[pos].append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
    width_table = {
        f"{jaw}_{pos}": {"n": len(v), "measured_median_mm": round(float(np.median(v)), 2),
                         "textbook_mm": expected_width((10 if jaw == "upper" else 30) + pos)}
        for (jaw, pos), v in sorted(widths.items())
    }
    return {
        "scans": n_scans,
        "keypoints": len(surface),
        "distance_to_surface_mm": {"median": round(float(np.median(surface)), 4), "p99": round(float(np.percentile(surface, 99)), 4)},
        "per_tooth_counts_median": {str(pos): {c: float(np.median(v[c])) for c in CLASSES} for pos, v in sorted(counts.items())},
        "mesiodistal_width": width_table,
        "tooth_frame_mesial_error_deg": {str(pos): {"n": len(v), "median": round(float(np.median(v)), 1), "p90": round(float(np.percentile(v, 90)), 1)}
                                         for pos, v in sorted(frame_err.items())},
    }


def main() -> None:
    from .datasets import iter_teeth3ds, teeth3ds_members

    p = argparse.ArgumentParser(description="解析和统计 3DTeethLand 关键点")
    p.add_argument("--zips", type=Path, required=True, help="Teeth3DS 压缩包目录（含 3DTeethLand_landmarks_annotation）")
    p.add_argument("--limit", type=int)
    p.add_argument("--out", type=Path, default=Path("eval_keypoints"))
    args = p.parse_args()
    members = teeth3ds_members(args.zips)
    stems = sorted(s for s, m in members.items() if "kpt" in m and "obj" in m)[: args.limit]
    print(f"有关键点的扫描 {len(stems)} 个", flush=True)
    result = analyze(iter_teeth3ds(args.zips, stems, members))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "keypoints_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
