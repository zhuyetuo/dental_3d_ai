"""CrossTooth（CVPR 2025）点云分支的批量推理，在 CrossTooth 自己的虚拟环境里运行（crosstooth_eval.sh 调用）。

作者公开的 predict.py 只处理一个已经预处理好的 ply、输出彩色网格。这里批量处理我们导出的 OBJ，输出逐顶点类别号：
1. 按文件顺序读出全部 v 行（含没有被三角形引用的孤立顶点，Teeth3DS 约 7%）；
2. 简化到 16000 个三角形以内。作者用的是没有公开的 selective_downsample.exe（按曲率保留细节），这里只能用普通的二次误差简化
   （fast-simplification），**和原文流程不同，分数可能低于论文**；
3. 用作者的 ToothData 预处理（三角形中心 + 法向，归一化）和 PointTransformerSeg38 预测每个三角形的类别（0–18）；
4. 每个原始顶点取最近的简化后三角形中心的类别，写成 <扫描名>.txt（一行一个，与 OBJ 的 v 行对应）。

    python crosstooth_predict.py --repo third_party/CrossTooth_CVPR2025 --data-dir data/fair_test --out runs/crosstooth/txt
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

NUM_FACES = 16000


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                verts.append(line.split()[1:4])
            elif line.startswith("f "):
                idx = [int(t.split("/")[0]) for t in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                faces += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def simplify(verts: np.ndarray, faces: np.ndarray, target: int = NUM_FACES) -> tuple[np.ndarray, np.ndarray]:
    """二次误差简化到不超过 target 个三角形（作者的数据加载只取前 target 行，多出来的会被丢掉）。"""
    import fast_simplification

    used = np.unique(faces)
    remap = np.full(len(verts), -1)
    remap[used] = np.arange(len(used))
    v, f = verts[used], remap[faces]
    if len(f) <= target:
        return v, f
    goal = target
    for _ in range(6):
        out_v, out_f = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=1 - goal / len(f))
        if len(out_f) <= target:
            return out_v.astype(np.float64), out_f.astype(np.int64)
        goal = int(goal * target / len(out_f)) - 1
    raise RuntimeError(f"简化后仍有 {len(out_f)} 个三角形")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--weights", type=Path, help="默认 <repo>/models/PTv1/point_best_model.pth")
    args = p.parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))

    import torch
    import trimesh
    from scipy.spatial import cKDTree

    from dataset.data import ToothData
    from models.PTv1.point_transformer_seg import PointTransformerSeg38

    device = torch.device("cuda")
    model = PointTransformerSeg38(in_channels=6, num_classes=17 + 2, pretrain=False, add_cbl=False, enable_pic_feat=False).to(device)
    model.load_state_dict(torch.load(args.weights or repo / "models/PTv1/point_best_model.pth", map_location=device))
    model.eval()
    loader = ToothData(argparse.Namespace(num_points=NUM_FACES, sample_points=NUM_FACES))

    args.out.mkdir(parents=True, exist_ok=True)
    objs = sorted(args.data_dir.rglob("*.obj"))
    timing = []
    for i, obj in enumerate(objs, 1):
        dest = args.out / f"{obj.stem}.txt"
        if dest.exists():
            continue
        t0 = time.time()
        verts, faces = read_obj(obj)
        sv, sf = simplify(verts, faces)
        with tempfile.TemporaryDirectory() as tmp:
            ply = Path(tmp) / f"{obj.stem}.ply"
            trimesh.Trimesh(sv, sf, process=False).export(ply)
            np.random.seed(1)
            torch.manual_seed(1)
            pointcloud, point_coords, face_info = loader.get_by_name(str(ply))
        with torch.no_grad():
            x = pointcloud.unsqueeze(0).to(device).permute(0, 2, 1).contiguous()
            seg, _ = model(x)
            pred = seg.argmax(dim=1).squeeze(0).cpu().numpy()
        face_info = np.asarray(face_info).astype(np.int64)
        real = np.any(face_info != 0, axis=1)  # 补齐用的全零行不是真的三角形
        centers = np.asarray(point_coords)[face_info[real]].mean(axis=1)
        _, nearest = cKDTree(centers).query(verts)
        labels = pred[real][nearest]
        np.savetxt(dest, labels.astype(np.int64), fmt="%d")
        timing.append(time.time() - t0)
        print(f"[{i}/{len(objs)}] {obj.stem} {len(faces)} → {len(sf)} 个三角形，{timing[-1]:.1f} 秒", flush=True)
    if timing:
        print(f"平均每例 {np.mean(timing):.2f} 秒（含简化）")


if __name__ == "__main__":
    main()
