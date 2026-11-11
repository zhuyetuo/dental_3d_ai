#!/bin/bash
# 训练学习式缺失牙生成模型（模板变形网络，开口穹顶模板）并评估。用法：bash scripts/gpu/train_toothgen.sh
# 需要先跑 get_data.sh（data/drive/raw/Teeth3DS 和 data/fair_test）。可用环境变量调整：EPOCHS（默认 60）、BATCH（默认 32）。
# models/generation/toothgen_open.pt 已存在（仓库自带）时接着训练：LR（默认 3e-4）；FRESH=1 从头训练。
# 输出：models/generation/toothgen_open.pt、runs/toothgen_open/、runs/gen_compare/{mirror,learned}/generation_summary.json
set -euo pipefail
ulimit -n "$(ulimit -Hn)"
cd "$(dirname "$0")/../../ml"
Z=../data/drive/raw/Teeth3DS
S=$Z/Teeth3DS+/Teeth3DS_train_test_split
mkdir -p ../runs/toothgen_open
python -m dental_ai.generation.learned cache --zips $Z --split $S/training_lower.txt $S/training_upper.txt --out ../data/gen_cache/train --workers "$(( $(nproc) > 48 ? 48 : $(nproc) ))"
python -m dental_ai.generation.learned cache --data-dir ../data/fair_test --out ../data/gen_cache/fair_test --workers "$(( $(nproc) > 48 ? 48 : $(nproc) ))"
M=../models/generation/toothgen_open.pt
RESUME=()
# 已有模型时接着训练（学习率调小；只有验证集上更好才覆盖原文件）。FRESH=1 从头训练
if [ "${FRESH:-0}" != 1 ] && [ -f "$M" ]; then RESUME=(--init "$M" --lr "${LR:-3e-4}"); echo "从 $M 接着训练"; fi
python -u -m dental_ai.generation.learned train --cache ../data/gen_cache/train --out "$M" \
  --template dome --epochs "${EPOCHS:-60}" --batch-size "${BATCH:-32}" "${RESUME[@]}" | tee -a ../runs/toothgen_open/train.log
python -m dental_ai.generation.learned eval --model ../models/generation/toothgen_open.pt --cache ../data/gen_cache/fair_test --out ../runs/toothgen_open/eval_fair_test
# 同口径对比：模拟缺牙后跑完整流程（生成、邻接调整、封底），与真牙比较。先是原来的对侧镜像 / 模板，再是学习模型
W="$(( $(nproc) > 32 ? 32 : $(nproc) ))"
python -m dental_ai.eval_generation --data-dir ../data/fair_test --workers "$W" --out ../runs/gen_compare/mirror
EVAL_GEN_MODEL="$(realpath ../models/generation/toothgen_open.pt)" GEN_DEVICE=cpu OMP_NUM_THREADS=2 \
  python -m dental_ai.eval_generation --data-dir ../data/fair_test --workers "$W" --out ../runs/gen_compare/learned
echo "结果：runs/toothgen_open/eval_fair_test/summary.json、runs/gen_compare/*/generation_summary.json"
