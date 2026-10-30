"""生成合成口扫数据（网格 + Teeth3DS 格式标签），用于前后端联调。

    python scripts/make_synthetic.py --jaw lower --missing 36 --out samples/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))

from dental_ai.io import save_labels  # noqa: E402
from dental_ai.synthetic import synthetic_arch  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--jaw", choices=["upper", "lower"], default="lower")
p.add_argument("--missing", type=int, nargs="*", default=[36])
p.add_argument("--out", default="samples")
args = p.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
mesh, labels = synthetic_arch(args.jaw, missing=tuple(args.missing))
mesh.export(out / f"synthetic_{args.jaw}.obj")
save_labels(out / f"synthetic_{args.jaw}.json", labels, args.jaw, patient_id="synthetic")
print(f"已写入 {out}/synthetic_{args.jaw}.obj 和 .json")
