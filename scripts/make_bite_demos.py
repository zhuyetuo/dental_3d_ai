"""上下颌成对的本地示例（demo/local/，不提交）：咬合对齐的口扫里去掉一颗牙，另一个颌当对颌。

    python scripts/make_bite_demos.py --zip ~/zyt/dental_data/Bite2Text.zip --labels data/bite_labels

数据：Bite2Text（CC BY-NC-SA，需要注册下载），不能放进公开仓库，所以只写到 demo/local/（.gitignore 已排除）。
牙的范围用分割模型自动标（--labels 是 learned.py cache-bite / eval_crown 留下的 <患者>_<颌>.npy；没有的会现算，需要 --seg-model）。
仓库自带的示例（Teeth3DS）上下颌不在咬合位，所以都是单颌的。
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

# (患者, 颌, 去掉的牙)：评分卡用的患者（不在生成模型的训练数据里）
DEMOS = [("F2005", "lower", 36), ("F5212", "upper", 26), ("F5051", "lower", 45)]


def main() -> None:
    from dental_ai.eval_crown import _labels, _load

    p = argparse.ArgumentParser(description="上下颌成对的本地示例（Bite2Text）")
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--seg-model", default=str(ROOT / "models/segmentation/deep.pt"))
    args = p.parse_args()
    out = ROOT / "demo/local"
    (out / "scans").mkdir(parents=True, exist_ok=True)
    listing = out / "demos.json"
    demos = [d for d in (json.loads(listing.read_text()) if listing.exists() else []) if not d["id"].startswith("bite-")]
    for pid, jaw, fdi in DEMOS:
        other = "upper" if jaw == "lower" else "lower"
        for j in (jaw, other):
            mesh = _load(str(args.zip), pid, j)
            dest = out / "scans" / f"{pid}_{j}.obj.gz"
            if not dest.exists():
                with gzip.open(dest, "wt", compresslevel=5) as f:
                    f.write(mesh.export(file_type="obj", include_normals=False, include_color=False, include_texture=False))
            if j == jaw:
                labels = _labels(mesh, pid, j, args.seg_model, args.labels)
                assert len(labels) == len(mesh.vertices) and fdi in set(labels.tolist()), (pid, j, fdi)
                with gzip.open(out / "scans" / f"{pid}_{j}.json.gz", "wt") as f:
                    json.dump({"id_patient": pid, "jaw": j, "labels": labels.astype(int).tolist()}, f)
        name = "下颌" if jaw == "lower" else "上颌"
        demos.append({
            "id": f"bite-{pid}-{fdi}", "title": f"上下颌咬合：{name}缺 {fdi}",
            "description": f"咬合对齐的上下颌口扫，{name}去掉 {fdi}；生成时看着对颌，三维视图里对颌半透明显示（Bite2Text，分割模型自动标牙）",
            "scan": f"{pid}_{jaw}", "antagonist": f"{pid}_{other}", "missing": [fdi], "labels": True, "jaw": jaw,
        })
        print("ok", demos[-1]["id"], flush=True)
    listing.write_text(json.dumps(demos, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
