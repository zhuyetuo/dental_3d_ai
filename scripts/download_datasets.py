"""下载公开数据集，整理成 Teeth3DS 目录结构（<obj>/<病例>/<病例>_<颌>.obj 与 <gt>/<病例>/<病例>_<颌>.json）。

    python scripts/download_datasets.py teeth3ds-sample   # 官方仓库里的 1 个真实口扫 + 标注（约 12 MB），冒烟测试用
    python scripts/download_datasets.py teeth3ds-splits   # 官方训练 / 测试划分文件
    python scripts/download_datasets.py teeth3ds          # 完整 Teeth3DS（OSF，6 个分卷，几十 GB）

默认下载到 data/（已在 .gitignore 中，数据不进 git）。

许可：Teeth3DS 是 CC BY-NC-ND 4.0，只能用于非商业研究和教学；训练或评估商业模型需要向 contact@udini.ai 申请商业授权。
不要把数据或由它派生的文件（如模板）提交到仓库或分发。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "abenhamadou/3DTeethSeg22_challenge"
COMMIT = "6220bfa6c26a1f24ab5f19ee1bc918e10a5d425e"  # 固定版本，保证可复现
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}"

SAMPLE = {  # 仓库中 refrence_algorithm_submission/test 下的示例口扫（官方拼写就是 refrence）
    "obj": ("refrence_algorithm_submission/test/0EJBIPTC_lower.obj", "b824f6822f4a6ada296eef6869e9341fa1e69ad6cd14862b53572198ae5e7a76"),
    "json": ("refrence_algorithm_submission/test/0EJBIPTC_lower.json", "b9f5279e05284c6a5d19aac2874ecaeec82122b480818b8a5f54a0be7e814e90"),
}
SPLITS = [
    "dataset/3DTeethSeg_challenge_split/public-training-set-1.txt",
    "dataset/3DTeethSeg_challenge_split/public-training-set-2.txt",
    "dataset/3DTeethSeg_challenge_split/private-testing-set.txt",
    "dataset/Teeth3DS_split/training_lower.txt",
    "dataset/Teeth3DS_split/training_upper.txt",
    "dataset/Teeth3DS_split/testing_lower.txt",
    "dataset/Teeth3DS_split/testing_upper.txt",
]
OSF_NODE = "xctdy"


def fetch(url: str, dest: Path, sha256: str | None = None) -> Path:
    if dest.exists() and (sha256 is None or _sha256(dest) == sha256):
        print(f"已存在 {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"下载 {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    if sha256 and _sha256(tmp) != sha256:
        tmp.unlink()
        raise SystemExit(f"校验失败：{url}")
    tmp.rename(dest)
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def teeth3ds_sample(out: Path) -> None:
    case, stem = "0EJBIPTC", "0EJBIPTC_lower"
    fetch(f"{RAW}/{SAMPLE['obj'][0]}", out / "obj" / case / f"{stem}.obj", SAMPLE["obj"][1])
    fetch(f"{RAW}/{SAMPLE['json'][0]}", out / "gt" / case / f"{stem}.json", SAMPLE["json"][1])
    print(f"完成：{out}（--data-dir {out / 'obj'} --gt-dir {out / 'gt'}）")


def teeth3ds_splits(out: Path) -> None:
    for path in SPLITS:
        fetch(f"{RAW}/{path}", out / Path(path).parent.name / Path(path).name)


def _osf_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def _osf_files(url: str):
    """递归列出 OSF 存储中的文件：(相对路径, 下载地址, 大小)。"""
    while url:
        page = _osf_json(url)
        for item in page["data"]:
            attrs = item["attributes"]
            if attrs["kind"] == "folder":
                yield from _osf_files(item["relationships"]["files"]["links"]["related"]["href"])
            else:
                yield attrs["materialized_path"].lstrip("/"), item["links"]["download"], attrs.get("size")
        url = page["links"].get("next")


def teeth3ds_full(out: Path, extract: bool) -> None:
    """完整数据集：列出 OSF 项目（含子项目）中的全部文件并下载，可选解压。

    未在本仓库的开发环境中实测（那里访问不了 osf.io）；分卷名称或目录结构变化时，按官方 README 手动下载解压到同样结构即可。"""
    nodes = [OSF_NODE] + [c["id"] for c in _osf_json(f"https://api.osf.io/v2/nodes/{OSF_NODE}/children/")["data"]]
    archives = []
    for node in nodes:
        for rel, url, size in _osf_files(f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/"):
            print(f"{rel}  {size or '?'} 字节")
            archives.append(fetch(url, out / "raw" / rel))
    if not extract:
        return
    for path in archives:
        if path.suffix == ".zip":
            print(f"解压 {path.name}")
            with zipfile.ZipFile(path) as z:
                z.extractall(out / "extracted")
    print(f"解压到 {out / 'extracted'}。检查目录结构后，把 OBJ 目录和标注目录分别传给 --data-dir / --gt-dir")


def main() -> None:
    p = argparse.ArgumentParser(description="下载公开牙科数据集（遵守各数据集许可）")
    p.add_argument("dataset", choices=["teeth3ds-sample", "teeth3ds-splits", "teeth3ds"])
    p.add_argument("--out", type=Path, help="输出目录（默认 data/<数据集>）")
    p.add_argument("--no-extract", action="store_true", help="完整数据集只下载不解压")
    args = p.parse_args()
    out = args.out or ROOT / "data" / args.dataset.replace("-", "_")
    print("Teeth3DS 许可：CC BY-NC-ND 4.0，仅限非商业研究；商用需另行授权（contact@udini.ai）。", file=sys.stderr)
    if args.dataset == "teeth3ds-sample":
        teeth3ds_sample(out)
    elif args.dataset == "teeth3ds-splits":
        teeth3ds_splits(out)
    else:
        teeth3ds_full(out, extract=not args.no_extract)


if __name__ == "__main__":
    main()
