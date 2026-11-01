"""颈缘线检测评估：真实预备体（Alsheghri 等公开的 54 个切牙代型，MIT 许可）。

数据：https://github.com/intellident-ai/teethPreparationData ，VTP 网格，每个三角形 Label = 1（颈缘线以上的预备体）
/ 0（颈缘线以下的代型），两者交界就是技工确认的颈缘线。论文：arXiv 2507.22859。

我们的检测器（margin/detect.py）从分割给出的初始线出发，吸附到附近的凸棱上。这些代型上没有牙龈、没有分割结果，
所以把真值边界向上（预备体内）或向下（代型上）平移 offset mm 当作初始线，模拟分割误差，看检测器能不能吸回真值：

    python -m dental_ai.eval_margin --data-dir "../data/prep_margin/test Data" --out ../runs/eval_margin

指标：检测线上的点到真值线的距离（mm，平均 / 95% / 最大），以及比 0.2 mm（论文用的“与技工一致”阈值）更近的点的比例。
初始线自身的误差一起列出，作为“什么都不做”的对照。读 VTP 需要 `pip install vtk`。
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .margin.cervical import boundary_loops
from .margin.detect import detect_margin, edge_sharpness, resample_loop

OFFSETS = (0.0, 0.3, -0.3, 0.6, -0.6, 1.0, -1.0)
PREP = 1  # 用 1 当作“牙号”，0 = 代型 / 背景


def load_vtp(path: Path) -> tuple[trimesh.Trimesh, np.ndarray]:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    points = vtk_to_numpy(data.GetPoints().GetData()).astype(float)
    faces = vtk_to_numpy(data.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    labels = vtk_to_numpy(data.GetCellData().GetArray("Label")).astype(int)
    return trimesh.Trimesh(points, faces, process=False), labels


def truth_line(mesh: trimesh.Trimesh, face_label: np.ndarray, spacing: float = 0.02) -> np.ndarray:
    """真值颈缘线：Label=1 区域的最长边界环，按 spacing mm 加密（用来算点到线的距离）。"""
    prep = mesh.submesh([np.nonzero(face_label == PREP)[0]], append=True, repair=False)
    loop = prep.vertices[boundary_loops(prep)[0]]
    seg = np.roll(loop, -1, axis=0) - loop
    n = np.maximum(1, np.ceil(np.linalg.norm(seg, axis=1) / spacing).astype(int))
    return np.concatenate([loop[i] + np.linspace(0, 1, k, endpoint=False)[:, None] * seg[i] for i, k in enumerate(n)])


def shifted_labels(mesh: trimesh.Trimesh, face_label: np.ndarray, line: np.ndarray, up: np.ndarray, offset: float) -> np.ndarray:
    """顶点标签：把真值边界沿“上方向”平移 offset mm（正 = 往预备体里收，负 = 往代型上放）。

    每个顶点按到真值线最近点的高度差决定：高于“最近点高度 + offset”的算预备体。"""
    v = np.asarray(mesh.vertices)
    _, idx = cKDTree(line).query(v)
    above = (v - line[idx]) @ up
    vertex_prep = np.zeros(len(v), dtype=bool)
    np.logical_or.at(vertex_prep, mesh.faces.ravel(), np.repeat(face_label == PREP, 3))
    if offset == 0:
        return np.where(vertex_prep, PREP, 0)
    # 只改真值线附近 3 mm 内的顶点，避免切端 / 代型底部被误改
    near = np.linalg.norm(v - line[idx], axis=1) < max(3.0, 3 * abs(offset))
    moved = np.where(near, above > offset, vertex_prep)
    return np.where(moved, PREP, 0)


def line_errors(points: np.ndarray, truth_tree: cKDTree) -> dict:
    d = truth_tree.query(points)[0]
    return {"mean": float(d.mean()), "p95": float(np.percentile(d, 95)), "max": float(d.max()), "within_0.2": float((d < 0.2).mean())}


def evaluate_case(path: Path, detect_kwargs: dict | None = None) -> list[dict]:
    mesh, face_label = load_vtp(path)
    line = truth_line(mesh, face_label)
    tree = cKDTree(line)
    prep_faces = face_label == PREP
    up = (mesh.face_normals[prep_faces] * mesh.area_faces[prep_faces, None]).sum(axis=0)
    up /= np.linalg.norm(up)
    sharp = edge_sharpness(mesh)
    rows = []
    for offset in OFFSETS:
        row = {"case": path.stem.replace("_labeledPrepFinal", ""), "offset": offset}
        try:
            labels = shifted_labels(mesh, face_label, line, up, offset)
            res = detect_margin(mesh, labels, PREP, up=up, sharpness=sharp, **(detect_kwargs or {}))
            from .margin.cervical import cervical_line

            init = resample_loop(cervical_line(mesh, labels, PREP, smooth_iters=3), 120)
            row.update({f"init_{k}": v for k, v in line_errors(init, tree).items()})
            row.update({f"det_{k}": v for k, v in line_errors(res.points, tree).items()})
            row["confidence"] = res.confidence
        except Exception as exc:  # 跑完整批，单个失败记下来
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for offset in OFFSETS:
        ok = [r for r in rows if r["offset"] == offset and "det_mean" in r]
        if not ok:
            continue

        def med(k):
            return round(float(np.median([r[k] for r in ok])), 3)

        out[str(offset)] = {
            "cases": len(ok),
            "init_mean": med("init_mean"),
            "det_mean": med("det_mean"),
            "det_p95": med("det_p95"),
            "init_within_0.2": med("init_within_0.2"),
            "det_within_0.2": med("det_within_0.2"),
            "det_better": sum(r["det_mean"] < r["init_mean"] for r in ok),
            "det_mean_worst": round(max(r["det_mean"] for r in ok), 3),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="颈缘线检测评估（真实切牙预备体，模拟分割误差）")
    p.add_argument("--data-dir", type=Path, required=True, action="append", help="可以给多次（训练集 + 测试集）")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("eval_margin"))
    p.add_argument("--search-radius", type=float, help="覆盖 detect_margin 的默认值")
    p.add_argument("--max-height-diff", type=float, help="覆盖 detect_margin 的默认值")
    args = p.parse_args()
    kw = {k: v for k, v in (("search_radius", args.search_radius), ("max_height_diff", args.max_height_diff)) if v is not None}
    files = sorted(f for d in args.data_dir for f in d.glob("*.vtp"))
    print(f"找到 {len(files)} 个预备体", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for part in pool.map(evaluate_case, files, [kw] * len(files)):
            rows += part
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "margin_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    summary = summarize(rows)
    (args.out / "margin_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    errors = [r for r in rows if "error" in r]
    if errors:
        print(f"{len(errors)} 个失败，例如：{errors[0]}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
