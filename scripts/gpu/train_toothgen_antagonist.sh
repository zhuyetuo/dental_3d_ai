#!/bin/bash
# 训练带对颌输入的缺失牙生成模型（在 train_toothgen.sh 的模型基础上接着训）。用法：bash scripts/gpu/train_toothgen_antagonist.sh
# 需要：train_toothgen.sh 跑过（data/gen_cache/train、models/generation/toothgen_open.pt）、models/segmentation/deep.pt、
#       咬合对齐的上下颌口扫 BITE_ZIP（默认 ~/zyt/dental_data/Bite2Text.zip，每位患者 <id>/ios/ios_{upper,lower}.stl）。
# 环境变量：EPOCHS（默认 20）、EXCLUDE（一个目录，里面 <患者>_<颌>.npy 出现过的患者不参加训练，如评分卡的 runs/crown_quality/labels）。
# 输出：runs/toothgen_ant/toothgen_ant.last.pt（确认效果后再覆盖 models/generation/toothgen_open.pt）、runs/toothgen_ant/eval_*/summary.json
set -euo pipefail
ulimit -n "$(ulimit -Hn)"
cd "$(dirname "$0")/../../ml"
ZIP="${BITE_ZIP:-$HOME/zyt/dental_data/Bite2Text.zip}"
EX=()
[ -n "${EXCLUDE:-}" ] && EX=(--exclude "$EXCLUDE")
python -m dental_ai.generation.learned cache-bite --zip "$ZIP" --seg-model ../models/segmentation/deep.pt \
  --labels ../data/bite_labels "${EX[@]}" --out ../data/gen_cache/bite --workers "$(( $(nproc) > 24 ? 24 : $(nproc) ))"
# 每 18 位患者留 1 位做验证
mkdir -p ../data/gen_cache/bite_val ../runs/toothgen_ant
ls ../data/gen_cache/bite | sed 's/_\(upper\|lower\)\.npz//' | sort -u | awk 'NR % 18 == 0' | while read -r p; do mv ../data/gen_cache/bite/"${p}"_*.npz ../data/gen_cache/bite_val/; done
OLD=../models/generation/toothgen_open.pt
OUT=../runs/toothgen_ant/toothgen_ant.pt
python -u -m dental_ai.generation.learned train --cache ../data/gen_cache/train ../data/gen_cache/bite --template dome \
  --epochs "${EPOCHS:-20}" --batch-size 32 --init "$OLD" --lr 3e-4 --antagonist --refine --clear-weight 1.0 --out "$OUT" | tee ../runs/toothgen_ant/train.log
LAST=../runs/toothgen_ant/toothgen_ant.last.pt
E="python -m dental_ai.generation.learned eval --samples 0"
$E --model "$OLD" --cache ../data/gen_cache/bite_val --out ../runs/toothgen_ant/eval_bite_before
$E --model "$LAST" --cache ../data/gen_cache/bite_val --out ../runs/toothgen_ant/eval_bite
$E --model "$LAST" --cache ../data/gen_cache/bite_val --out ../runs/toothgen_ant/eval_bite_hidden --hide-antagonist
$E --model "$LAST" --cache ../data/gen_cache/fair_test --out ../runs/toothgen_ant/eval_fair
echo "评分卡：python -m dental_ai.eval_crown --zip $ZIP --seg-model ../models/segmentation/deep.pt --gen-model $LAST --out ../runs/crown_quality_new"
