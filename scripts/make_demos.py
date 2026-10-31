"""准备网页演示用的示例病例（demo/），随代码仓库一起提交。

    python scripts/make_demos.py --data-dir data/teeth3ds_test100
    python scripts/make_demos.py --local --poseidon-dir data/poseidon3d/test-complex   # 正畸困难病例，只在本地

--local：Poseidon3D 的正畸病例（拥挤、拔牙间隙等）写到 demo/local/（.gitignore 排除，不提交）。Poseidon3D 的许可还没核实，
仓库是公开的，不能放进去。导出 Poseidon3D 见 docs/datasets.md（python -m dental_ai.datasets poseidon3d ...）。

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

LOCAL_SCANS = ["P000011_upper", "P000002_upper", "P000041_lower", "P000100_upper", "P000037_upper"]
LOCAL_DEMOS = [
    {"id": "ortho-crowding-closed-space", "title": "正畸：拥挤 + 拔牙间隙关闭",
     "description": "Poseidon3D 困难病例：牙列拥挤，15 的拔牙间隙已关闭（识别为“间隙已关闭”，不生成），13 处有缺隙", "scan": "P000011_upper"},
    {"id": "ortho-extraction", "title": "正畸：拔牙病例（只剩 9 颗牙）",
     "description": "多处缺牙和游离端，看缺牙检测、编号和生成在困难病例上的表现", "scan": "P000002_upper"},
    {"id": "ortho-anterior-missing", "title": "下颌前牙大面积缺失",
     "description": "31–43 连续缺失、只剩 8 颗牙：连续缺失的位置分配", "scan": "P000041_lower"},
    {"id": "ortho-closed-gap", "title": "正畸：间隙已关闭",
     "description": "25 已拔除且间隙关闭，系统应识别为不需要修复", "scan": "P000100_upper"},
    {"id": "ortho-segment", "title": "正畸拥挤：只有网格，自动分割",
     "description": "带底座的正畸模型、牙列拥挤，不带标签，看自动分割和编号纠错", "scan": "P000037_upper",
     "labels": False, "requires": "segmenter"},
]

p = argparse.ArgumentParser()
p.add_argument("--data-dir", type=Path, help="导出的 Teeth3DS 扫描目录（<病例>/<病例>_<颌>.obj + .json）")
p.add_argument("--local", action="store_true", help="生成本地示例（Poseidon3D 正畸病例）到 demo/local/")
p.add_argument("--poseidon-dir", type=Path, default=ROOT / "data/poseidon3d/test-complex")
p.add_argument("--license", type=Path, default=ROOT / "data/drive/raw/Teeth3DS/Teeth3DS+/license.txt")
args = p.parse_args()

if args.local:
    out = ROOT / "demo" / "local"
    data_dir, scans, demos = args.poseidon_dir, LOCAL_SCANS, LOCAL_DEMOS
else:
    if not args.data_dir:
        p.error("需要 --data-dir（或 --local）")
    out = ROOT / "demo"
    data_dir, scans, demos = args.data_dir, SCANS, DEMOS
(out / "scans").mkdir(parents=True, exist_ok=True)
for stem in scans:
    case = stem.rsplit("_", 1)[0]
    for ext in ("obj", "json"):
        src = data_dir / case / f"{stem}.{ext}"
        with open(src, "rb") as f, gzip.open(out / "scans" / f"{stem}.{ext}.gz", "wb", compresslevel=9) as g:
            shutil.copyfileobj(f, g)  # 原样压缩，内容不改
for d in demos:
    d.setdefault("labels", True)
    d["jaw"] = d["scan"].rsplit("_", 1)[1]
(out / "demos.json").write_text(json.dumps(demos, ensure_ascii=False, indent=2))
if not args.local and args.license.exists():
    shutil.copy(args.license, out / "LICENSE-Teeth3DS.txt")
print("\n".join(f"{p.name} {p.stat().st_size / 1e6:.1f} MB" for p in sorted((out / "scans").iterdir())))
