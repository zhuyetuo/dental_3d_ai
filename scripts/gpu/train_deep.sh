#!/bin/bash
# 训练深度分割模型并评估。可用环境变量调整：EPOCHS（默认 60）、N_POINTS（默认 8000）、BATCH（默认 8）。
set -euo pipefail
cd "$(dirname "$0")/../../ml"
Z=../data/drive/raw/Teeth3DS
S=$Z/Teeth3DS+/Teeth3DS_train_test_split
mkdir -p ../runs/deep
python -m dental_ai.segmentation.deep cache --zips $Z --split $S/training_lower.txt $S/training_upper.txt \
  --extra-dir ../data/poseidon3d/train --workers "$(nproc)" --out ../data/deep_cache
python -m dental_ai.segmentation.deep train --cache ../data/deep_cache --epochs "${EPOCHS:-60}" --n-points "${N_POINTS:-8000}" \
  --batch-size "${BATCH:-8}" --out ../models/segmentation/deep.pt | tee ../runs/deep/train.log
for d in fair_test teeth3ds_test100 poseidon3d/test-complex; do
  name=$(basename $d)
  python -m dental_ai.segmentation.deep predict --model ../models/segmentation/deep.pt --data-dir ../data/$d --out ../runs/deep/pred_$name
  python -m dental_ai.evaluation --data-dir ../data/$d --gt-dir ../data/$d --pred-dir ../runs/deep/pred_$name --out ../runs/deep/eval_$name | tail -14
done
echo "结果：runs/deep/eval_*/summary.json"
