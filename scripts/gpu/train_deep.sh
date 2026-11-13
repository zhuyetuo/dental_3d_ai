#!/bin/bash
# 训练深度分割模型并评估。可用环境变量调整：EPOCHS（默认 60）、N_POINTS（默认 8000）、BATCH（默认 8）、
# SIM_MISSING（每个扫描生成几个模拟缺牙的变体，默认 2）、ARCH（从头训练时的网络结构，默认 dgcnn_attn；接着训练时跟着原模型）。
set -euo pipefail
ulimit -n "$(ulimit -Hn)"  # 训练时一次打开全部缓存文件（1350 个），超过默认上限 1024
cd "$(dirname "$0")/../../ml"
Z=../data/drive/raw/Teeth3DS
S=$Z/Teeth3DS+/Teeth3DS_train_test_split
mkdir -p ../runs/deep
python -m dental_ai.segmentation.deep cache --zips $Z --split $S/training_lower.txt $S/training_upper.txt \
  --extra-dir ../data/poseidon3d/train --workers "$(( $(nproc) > 96 ? 96 : $(nproc) ))" --simulate-missing "${SIM_MISSING:-2}" --simulate-closed "${SIM_CLOSED:-2}" --out ../data/deep_cache
M=../models/segmentation/deep.pt
RESUME=(--arch "${ARCH:-dgcnn_attn}")
# 已有模型（仓库自带）时接着训练（学习率调小；只有验证集上更好才覆盖原文件）。FRESH=1 从头训练
if [ "${FRESH:-0}" != 1 ] && [ -f "$M" ]; then RESUME=(--init "$M" --lr "${LR:-3e-4}"); echo "从 $M 接着训练"; fi
python -u -m dental_ai.segmentation.deep train --cache ../data/deep_cache --epochs "${EPOCHS:-60}" --n-points "${N_POINTS:-8000}" \
  --batch-size "${BATCH:-8}" --out "$M" "${RESUME[@]}" | tee -a ../runs/deep/train.log
for d in fair_test teeth3ds_test100 poseidon3d/test-complex; do
  name=$(basename $d)
  python -m dental_ai.segmentation.deep predict --model ../models/segmentation/deep.pt --data-dir ../data/$d --out ../runs/deep/pred_$name
  python -m dental_ai.evaluation --data-dir ../data/$d --gt-dir ../data/$d --pred-dir ../runs/deep/pred_$name --out ../runs/deep/eval_$name | tail -14
done
# 缺牙口扫：模拟缺牙后看会不会把缺牙区认成牙、缺牙检测对不对
python -m dental_ai.eval_missing_seg --model "$M" --data-dir ../data/fair_test --out ../runs/deep/missing --workers 8 | tail -22
echo "结果：runs/deep/eval_*/summary.json、runs/deep/missing/summary.json"
