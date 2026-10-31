"""模板库命令行。

从 Teeth3DS 格式的标注数据构建模板库：
    python -m dental_ai.templates build --data-dir <obj 目录> --gt-dir <标注目录> --out templates/
直接从 Teeth3DS 压缩包构建（不用解压，用训练集划分）：
    python -m dental_ai.templates build --zips <data_part_*.zip 所在目录> --split <training_lower.txt> --out templates/ --workers 4

导出内置参数化模板（查看用）：
    python -m dental_ai.templates export-builtin --out builtin_templates/
"""

import argparse
import json
from pathlib import Path

from ..evaluation import find_scans
from .library import build_library, export_builtin


def main() -> None:
    p = argparse.ArgumentParser(description="标准牙模板库")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="从标注数据构建模板库")
    b.add_argument("--data-dir", type=Path)
    b.add_argument("--gt-dir", type=Path)
    b.add_argument("--zips", type=Path, help="Teeth3DS 压缩包目录（代替 --data-dir/--gt-dir）")
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--split", type=Path, help="只用这些病例（如训练集划分文件）")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--min-vertices", type=int, default=200, help="顶点数少于此值的牙齿不参与（分割不完整）")
    e = sub.add_parser("export-builtin", help="导出内置参数化模板")
    e.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    if args.cmd == "build":
        if args.zips:
            from ..datasets import teeth3ds_zip_scans

            scans = teeth3ds_zip_scans(args.zips, args.split)
        elif args.data_dir and args.gt_dir:
            scans = find_scans(args.data_dir, args.gt_dir, args.split)
        else:
            p.error("需要 --zips，或 --data-dir 和 --gt-dir")
        if not scans:
            p.error("没有找到 OBJ 与标注能对上的扫描")
        index = build_library(scans, args.out, args.min_vertices, args.workers)
        print(json.dumps(index, ensure_ascii=False, indent=2))
    else:
        print("\n".join(export_builtin(args.out)))


if __name__ == "__main__":
    main()
