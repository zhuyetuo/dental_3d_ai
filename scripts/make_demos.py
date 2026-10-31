"""准备网页演示用的示例病例（demo/），随代码仓库一起提交。

    python scripts/make_demos.py --data-dir data/teeth3ds_test100

许可：Teeth3DS 是 CC BY-NC-ND 4.0。仓库是公开的，只能放**未经修改**的原始扫描（注明出处、附许可）；
“去掉几颗牙”“磨成预备体”这些修改版属于演绎作品，ND 不允许分发，所以只在 demos.json 里写配置，
由后端在本地加载时生成（backend/app/demos.py）。不要把修改过的扫描或 Poseidon3D（许可未确认）放进来。
"""

import argparse
import gzip
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCANS = ["01KTSH6A_lower", "016N696F_upper", "RXKYWHG6_lower"]
DEMOS = [
    {"id": "lower-missing-36", "title": "下颌缺 36：对侧镜像生成",
     "description": "完整下颌去掉 36，用对侧 46 镜像生成，底面贴合牙龈", "scan": "01KTSH6A_lower", "missing": [36]},
    {"id": "lower-prep-46", "title": "下颌 46 预备体：颈缘线 + 牙冠",
     "description": "46 模拟磨成肩台预备体，自动检测颈缘线并生成预备体牙冠（可在编辑器里改颈缘线）",
     "scan": "01KTSH6A_lower", "prepared": [46]},
    {"id": "upper-natural-gap-25", "title": "上颌真实缺牙 25",
     "description": "原始扫描本来就缺 25（缺隙约 5 mm），检测缺牙并生成", "scan": "016N696F_upper"},
    {"id": "upper-missing-16-26", "title": "上颌 16、26 都缺：标准牙模板",
     "description": "左右同名牙都缺，没有对侧可镜像，用标准牙模板生成（有真实模板库时用真实牙冠）",
     "scan": "016N696F_upper", "missing": [16, 26]},
    {"id": "lower-multiple-missing", "title": "下颌多颗缺牙",
     "description": "原始扫描缺 34（有缺隙）和两侧第二磨牙（游离端）", "scan": "RXKYWHG6_lower"},
    {"id": "lower-segment", "title": "只有网格：自动分割",
     "description": "不带标签上传，用分割模型（GPU 服务或 CPU 基线）自动分割和编号", "scan": "01KTSH6A_lower",
     "labels": False, "requires": "segmenter"},
]

p = argparse.ArgumentParser()
p.add_argument("--data-dir", type=Path, required=True, help="导出的 Teeth3DS 扫描目录（<病例>/<病例>_<颌>.obj + .json）")
p.add_argument("--license", type=Path, default=ROOT / "data/drive/raw/Teeth3DS/Teeth3DS+/license.txt")
args = p.parse_args()

out = ROOT / "demo"
(out / "scans").mkdir(parents=True, exist_ok=True)
for stem in SCANS:
    case = stem.rsplit("_", 1)[0]
    for ext in ("obj", "json"):
        src = args.data_dir / case / f"{stem}.{ext}"
        with open(src, "rb") as f, gzip.open(out / "scans" / f"{stem}.{ext}.gz", "wb", compresslevel=9) as g:
            shutil.copyfileobj(f, g)  # 原样压缩，内容不改
for d in DEMOS:
    d.setdefault("labels", True)
    d["jaw"] = d["scan"].rsplit("_", 1)[1]
(out / "demos.json").write_text(json.dumps(DEMOS, ensure_ascii=False, indent=2))
if args.license.exists():
    shutil.copy(args.license, out / "LICENSE-Teeth3DS.txt")
print("\n".join(f"{p.name} {p.stat().st_size / 1e6:.1f} MB" for p in sorted((out / "scans").iterdir())))
