#!/bin/bash
# 下载 FDI 16 数据集（3Shape / DTU，VF-Net 论文用的数据）：7762 颗上颌右侧第一磨牙（牙号 16）的单颗牙冠网格。
# 用法：bash scripts/gpu/get_data_fdi16.sh            （用下面默认的 Drive 文件）
#   或：bash scripts/gpu/get_data_fdi16.sh <Drive 文件链接或 ID>
# 也可以在仓库根目录的 .env 里用 FDI16_FILE 覆盖默认值。
#
# 内容（2026-11-11 在本地下载核对过）：
#   fdi16_dataset/{train,val,test}/<编号>_fdi16_mesh.ply                  单颗牙冠网格（约 7000 顶点，单位 mm，已对齐到同一坐标系，底部开口）
#   fdi16_dataset/{train,val,test}/<编号>_fdi16_point_cloud.ply           顶点 + 面片中点的点云，带法向
#   fdi16_dataset/{train,val,test}/<编号>_fdi16_curvature_edgedistance.dat float64：先是每个点的曲率，再是每个点到边界的测地距离
#   train 4844 / val 1465 / test 1453；压缩包 7160405142 字节（6.7 GB），解压后 7.9 GB。
# 许可：CC BY-NC-SA，仅限非商业用途；引用 Ye et al., "Variational Autoencoding of Dental Point Clouds", arXiv 2307.10895。
# 注意：只有单颗牙，没有邻牙和牙龈，不能直接用来训练分割模型或现在这个“看邻牙生成缺失牙”的模型；
#       适合做 16 / 26（镜像）的牙冠形状先验或细节模板库。
#
# 访问不了 Google 的机器（国内机房）：在能访问的电脑上下好，再传到 data/drive/raw/FDI16/fdi16_dataset.zip，然后重新运行本脚本。
#   官方源 https://data.dtu.dk/articles/dataset/3Shape_FDI_16_Meshes_from_Intraoral_Scans/23626650 （国内机房实测 403）
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; [ -f .env ] && . ./.env; set +a
# 默认是项目自己的共享文件（仓库所有者 2026-11-11 要求写进脚本）；参数或 .env 可以覆盖
SRC="${1:-${FDI16_FILE:-https://drive.google.com/file/d/12tH72Qh6bMDtrBmyzWD27ZWnYy0m0XWq}}"
ID=$(echo "$SRC" | sed -E 's#.*/file/d/##; s#.*[?&]id=##; s#[/?&].*##')
RAW=data/drive/raw/FDI16
Z=$RAW/fdi16_dataset.zip
OUT=data/fdi16
SIZE=7160405142
mkdir -p $RAW $OUT

zip_ok() { [ -f "$Z" ] && python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).namelist()" "$Z" 2>/dev/null; }

if zip_ok; then
  echo "压缩包已在 $Z，跳过下载"
elif ! curl -s -o /dev/null -m 15 https://drive.google.com; then
  echo "这台机器访问不了 drive.google.com。请在能访问的电脑上下载："
  echo "  https://drive.google.com/file/d/$ID/view"
  echo "再传到这台机器的 $(pwd)/$Z（例如 scp -P <端口> fdi16_dataset.zip root@<服务器>:$(pwd)/$Z），然后重新运行本脚本。"
  exit 1
else
  # 直链，可断点续传；大文件要带 confirm=t 跳过“无法扫描病毒”的提示页
  curl -L -C - --retry 5 --retry-delay 10 -o "$Z" "https://drive.usercontent.google.com/download?id=$ID&export=download&confirm=t" || true
  if ! zip_ok; then
    echo "直链没下完整（Drive 对大文件有下载次数限制时会返回提示页），改用 gdown"
    [ "$(stat -c %s "$Z" 2>/dev/null || echo 0)" -lt 1000000 ] && rm -f "$Z"
    python -c "import gdown" 2>/dev/null || python -m pip install -q gdown
    python -m gdown "$ID" -O "$Z" --continue
  fi
  zip_ok || { echo "下载失败：$Z 不是完整的压缩包。过一段时间再试，或手动下载后放到这个位置。"; exit 1; }
fi
[ "$(stat -c %s "$Z")" = "$SIZE" ] || echo "提示：压缩包大小和记录的 $SIZE 字节不一致（文件可能更新过）"

if [ "$(find $OUT -name '*_mesh.ply' 2>/dev/null | wc -l)" -ge 7762 ]; then
  echo "已解压到 $OUT，跳过"
else
  echo "== 解压到 $OUT（7.9 GB）"
  python - "$Z" "$OUT" <<'PY'
import sys, zipfile
from pathlib import Path
z, out = zipfile.ZipFile(sys.argv[1]), Path(sys.argv[2])
for info in z.infolist():  # 去掉最外层的 fdi16_dataset/
    rel = info.filename.split("/", 1)[1] if "/" in info.filename else info.filename
    if not rel or info.is_dir():
        continue
    dst = out / rel
    if dst.exists() and dst.stat().st_size == info.file_size:
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(z.read(info))
PY
fi
for s in train val test; do echo "$s: $(find $OUT/$s -name '*_mesh.ply' | wc -l) 颗牙"; done
echo "数据就绪：$OUT/{train,val,test}（预期 4844 / 1465 / 1453）"
