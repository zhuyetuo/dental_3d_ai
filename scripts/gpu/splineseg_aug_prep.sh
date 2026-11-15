#!/bin/bash
# 微调数据：导出（CPU 40 进程）→ TeethSplineSeg 预处理（40 份并行，强制 CPU，不抢正在训练的 GPU）
cd ~/zyt/dental_3d_ai && . .venv/bin/activate
s=$(date +%s)
CUDA_VISIBLE_DEVICES= python scripts/gpu/splineseg_aug_export.py --workers 40 2>&1 | tail -3
echo "导出用时 $(( $(date +%s) - s )) 秒；$(du -sh data/splineseg_aug | cut -f1)"
SRC=$(realpath data/splineseg_aug)
ROOT=$(realpath third_party/TeethSplineSeg)/data/aug
SS=third_party/TeethSplineSeg
PY=$PWD/.venv-splineseg/bin/python
CACHE=processed_32000_points_7dim_arch_gsn_jawtype
K=40
W=/tmp/ssaug
rm -rf $W; mkdir -p $W "$ROOT/$CACHE"
s=$(date +%s)
for i in $(seq 0 $((K - 1))); do
  mkdir -p $W/src$i
  for jaw in lower upper; do
    ln -sfn "$SRC/$jaw" "$W/src$i/$jaw"
    cat "$SRC/training_$jaw.txt" "$SRC/testing_$jaw.txt" | awk -v i=$i -v k=$K 'NF && (NR - 1) % k == i' | while read n; do
      [ -f "$ROOT/$CACHE/data_$n.pt" ] || echo "$n"
    done > $W/src$i/testing_$jaw.txt
    : > $W/src$i/training_$jaw.txt
  done
  (cd $SS && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 $PY prepare_dataset.py --source $W/src$i --root $W/root$i > $W/log$i 2>&1; cp $W/root$i/$CACHE/data_*.pt "$ROOT/$CACHE"/ 2>/dev/null || true) &
done
wait
echo "预处理（$K 份，CPU）用时 $(( $(date +%s) - s )) 秒，共 $(ls "$ROOT/$CACHE" | grep -c '^data_') 个"
grep -h -i "failed" $W/log* | head -5
# 训练要的划分文件和 raw 链接
ln -sfn "$SRC" "$ROOT/raw"
cp "$SRC"/training_*.txt "$SRC"/testing_*.txt "$ROOT/" 2>/dev/null || true
echo AUG_PREP_DONE
