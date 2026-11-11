#!/bin/bash
# 不经过 Google 下载数据集并导出（服务器访问不了 Google Drive 时用，不需要任何密钥）。用法：bash scripts/gpu/get_data_osf.sh
# Teeth3DS 来自 OSF（osf.io/xctdy，CC BY-NC-ND 4.0，约 8.9 GB），Poseidon3D 来自 Zenodo（records/15608906，约 1 GB）。
# 文件摆成 get_data.sh 期望的结构（data/drive/raw/...），然后调用 get_data.sh 导出（它发现数据已在就跳过 Drive）。
# 已下载完整的文件会跳过，可以重复运行。2026-11-11 在国内机房实测可用。
set -euo pipefail
cd "$(dirname "$0")/../.."
T=data/drive/raw/Teeth3DS
P=data/drive/raw/Poseidon3D
mkdir -p $T/Teeth3DS+/Teeth3DS_train_test_split $T/Teeth3DS+/3DTeethSeg22_challenge_train_test_split $P

get() {  # get <url> <dest> <size>：断点续传，大小对上才算完成
  local url=$1 dest=$2 size=$3
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$size" ]; then echo "已存在 $dest"; return; fi
  # 已有的文件不是本脚本下到一半的（没有 .part 标记）而大小又不对：来源不同，重新下
  if [ -f "$dest" ] && [ ! -f "$dest.part" ]; then echo "大小不符，重新下载 $dest"; rm -f "$dest"; fi
  touch "$dest.part"
  for i in 1 2 3 4 5 6 7 8; do
    curl -sS -L -C - --retry 5 --retry-delay 5 -o "$dest" "$url" || true
    [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$size" ] && { rm -f "$dest.part"; echo "完成 $dest"; return; }
    [ -f "$dest" ] && [ "$(stat -c %s "$dest")" -gt "$size" ] && rm -f "$dest"
    echo "重试 $dest ($i)"; sleep 5
  done
  echo "下载失败 $dest"; return 1
}

get_segmented() {  # get_segmented <url> <dest> <size>：分 16 段并行（Zenodo 单连接只有几百 KB/s）
  local url=$1 dest=$2 size=$3 n=16
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$size" ]; then echo "已存在 $dest"; return; fi
  local ch=$(( (size + n - 1) / n )) seg="$dest.seg" pids=()
  mkdir -p "$seg"
  for i in $(seq 0 $((n - 1))); do
    local s=$((i * ch)) e=$((i * ch + ch - 1)); [ $e -ge $size ] && e=$((size - 1))
    local part="$seg/p$(printf %02d $i)"
    ( for t in 1 2 3 4 5 6; do
        curl -sS -L --retry 5 -r $s-$e -o "$part" "$url" || true
        [ "$(stat -c %s "$part" 2>/dev/null)" = "$((e - s + 1))" ] && exit 0
      done; exit 1 ) & pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  cat "$seg"/p* > "$dest" && rm -rf "$seg"
  [ "$(stat -c %s "$dest")" = "$size" ] || { echo "下载失败 $dest"; return 1; }
  echo "完成 $dest"
}

S=$T/Teeth3DS+/Teeth3DS_train_test_split
C=$T/Teeth3DS+/3DTeethSeg22_challenge_train_test_split
get https://osf.io/download/zkge5/ $S/training_lower.txt 9000
get https://osf.io/download/zs5cg/ $S/training_upper.txt 9000
get https://osf.io/download/snwye/ $S/testing_lower.txt 4500
get https://osf.io/download/c8g6u/ $S/testing_upper.txt 4500
get https://osf.io/download/pebtc/ $C/public-training-set-1.txt 9000
get https://osf.io/download/jd2fc/ $C/public-training-set-2.txt 9000
get https://osf.io/download/x7rfj/ $C/private-testing-set.txt 9000
get https://osf.io/download/9dutn/ $T/Teeth3DS+/license.txt 19127
pids=()
get https://osf.io/download/qhprs/ $T/data_part_1.zip 1517213360 & pids+=($!)
get https://osf.io/download/4pwnr/ $T/data_part_2.zip 1490597289 & pids+=($!)
get https://osf.io/download/frwdp/ $T/data_part_3.zip 1304892240 & pids+=($!)
get https://osf.io/download/2arn4/ $T/data_part_4.zip 1285490895 & pids+=($!)
get https://osf.io/download/xrz5f/ $T/data_part_5.zip 1340868934 & pids+=($!)
get https://osf.io/download/23hgq/ $T/data_part_6.zip 1379614841 & pids+=($!)
get https://osf.io/download/u83ad/ $T/data_part_7.zip 577966827 & pids+=($!)
get_segmented https://zenodo.org/api/records/15608906/files/poseidon3d.zip/content $P/poseidon3d.zip 1029192163 & pids+=($!)
for p in "${pids[@]}"; do wait $p; done
for z in $T/data_part_*.zip $P/poseidon3d.zip; do python -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).namelist()" "$z"; done
echo "下载完成："; du -sh data/drive/raw
# 导出各个测试集（get_data.sh 发现数据已在就跳过 Drive）。SKIP_EXPORT=1 只下载不导出
[ "${SKIP_EXPORT:-0}" = 1 ] || bash scripts/gpu/get_data.sh
