#!/bin/bash
# 从 Google Drive 共享文件夹下载数据集并导出，不需要 API 密钥（文件夹要设成“知道链接的任何人可查看”）。
# 用法（链接或文件夹 ID 都可以）：
#   bash scripts/gpu/get_data_drive.sh <Teeth3DS 文件夹链接> <Poseidon3D 文件夹链接>
# 也可以写在仓库根目录的 .env 里（TEETH3DS_FOLDER、POSEIDON3D_FOLDER），之后直接 bash scripts/gpu/get_data_drive.sh。
#
# 两种获取方式自动衔接：先走 Drive（gdown）；Drive 连不上、没给链接、或有文件没下全时，
# 剩下的由 get_data_osf.sh 从 OSF / Zenodo 官方源补齐（已下好的文件按大小核对后跳过），最后导出。
# 所以国内机房（访问不了 Google）和境外机房都用这一条命令。
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; [ -f .env ] && . ./.env; set +a
TEETH="${1:-${TEETH3DS_FOLDER:-}}"
POSEIDON="${2:-${POSEIDON3D_FOLDER:-}}"
T=data/drive/raw/Teeth3DS
P=data/drive/raw/Poseidon3D
mkdir -p $T $P

folder_id() {  # 链接（.../folders/<ID>?...）或 ID → ID
  echo "$1" | sed -E 's#.*/folders/##; s#[/?].*##'
}

flatten() {  # gdown 有的版本会多套一层同名目录：<dir>/<名字>/... → <dir>/...
  local dir=$1 name=$2
  if [ -d "$dir/$name" ]; then
    cp -al "$dir/$name/." "$dir/" 2>/dev/null || cp -a "$dir/$name/." "$dir/"
    rm -rf "${dir:?}/$name"
  fi
}

if [ -z "$TEETH" ] && [ -z "$POSEIDON" ]; then
  echo "没有给 Drive 文件夹链接，直接用 OSF / Zenodo"
elif ! curl -s -o /dev/null -m 15 https://drive.google.com; then
  echo "这台机器访问不了 drive.google.com，改用 OSF / Zenodo"
else
  python -c "import gdown" 2>/dev/null || python -m pip install -q gdown
  if [ -n "$TEETH" ]; then
    echo "== Drive：Teeth3DS"
    python -m gdown --folder "https://drive.google.com/drive/folders/$(folder_id "$TEETH")" -O $T --remaining-ok --continue \
      || echo "Teeth3DS 没有全部下完（Drive 对大文件有下载次数限制），缺的从 OSF 补"
    flatten $T Teeth3DS
  fi
  if [ -n "$POSEIDON" ]; then
    echo "== Drive：Poseidon3D"
    python -m gdown --folder "https://drive.google.com/drive/folders/$(folder_id "$POSEIDON")" -O $P --remaining-ok --continue \
      || echo "Poseidon3D 没有下完，从 Zenodo 补"
    flatten $P Poseidon3D
  fi
  # 没下完整的压缩包删掉（Drive 限流时会存下一个 HTML 提示页），让下一步从官方源重新下
  for z in $T/data_part_*.zip $P/poseidon3d.zip; do
    [ -f "$z" ] || continue
    python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).namelist()" "$z" 2>/dev/null || { echo "不完整，删除 $z"; rm -f "$z"; }
  done
fi
echo "== 核对并补齐（OSF / Zenodo），然后导出"
exec bash scripts/gpu/get_data_osf.sh
