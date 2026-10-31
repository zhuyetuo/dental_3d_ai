"""用真实口扫做演示数据：Teeth3DS 示例病例去掉几颗牙，上传到网页上看缺牙检测和生成效果。

    python scripts/download_datasets.py teeth3ds-sample
    python scripts/make_demo.py --missing 46 --out samples/

生成 samples/demo_lower.obj（网格）和 samples/demo_lower.json（标签）。在网页上同时上传这两个文件，
颌位选下颌，点“运行”。也可以只上传 .obj，再接分割服务。
许可：Teeth3DS 是 CC BY-NC-ND 4.0，只用于非商业研究演示，不要分发。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))

from dental_ai.io import load_labels, load_mesh, save_labels  # noqa: E402
from dental_ai.synthetic import simulate_missing  # noqa: E402

SAMPLE = ROOT / "data" / "teeth3ds_sample"

p = argparse.ArgumentParser(description="用 Teeth3DS 示例病例生成缺牙演示数据")
p.add_argument("--missing", type=int, nargs="*", default=[46], help="要去掉的牙号（下颌 31–47），不填则保留完整牙列")
p.add_argument("--out", default="samples")
args = p.parse_args()

obj = SAMPLE / "obj" / "0EJBIPTC" / "0EJBIPTC_lower.obj"
if not obj.exists():
    sys.exit("先下载示例病例：python scripts/download_datasets.py teeth3ds-sample")
mesh = load_mesh(obj)
labels, _ = load_labels(SAMPLE / "gt" / "0EJBIPTC" / "0EJBIPTC_lower.json")
if args.missing:
    mesh, labels = simulate_missing(mesh, labels, "lower", args.missing)

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
mesh.export(out / "demo_lower.obj")  # trimesh 写 OBJ 会保留孤立顶点，标签仍一一对应
save_labels(out / "demo_lower.json", labels, "lower", patient_id="0EJBIPTC")
print(f"已写入 {out / 'demo_lower.obj'} 和 {out / 'demo_lower.json'}（缺 {args.missing or '无'}）")
