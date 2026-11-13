#!/bin/bash
# 训练学习式颈缘线检测模型并评估，几何方法（margin/detect.py）的评估作对照。用法：bash scripts/gpu/train_margin.sh
# 数据（都自动下载；github.com 连不上时 codeload 往往还能用，所以直接下 zip）：
#   data/prep_margin/    teethPreparationData：52 个切牙代型（41 训练 / 11 测试），MIT
#   data/margin_public/  intellident-ai/public-datasets 的 Marginline detection：78 个代型（前牙到磨牙），仅限研究和教育用途；
#                        按患者划分成 split/train（63 个）和 split/test（15 个）
# 手里已经有这两个压缩包时，解压成上面的目录结构即可，不会重复下载。
# 可用环境变量调整：EPOCHS（默认 300）、BATCH（默认 8，约占 20 GB 显存；显存小就调小）、LR、FRESH=1 从头训练。
# 输出：models/margin/margin.pt、runs/margin/eval_test（切牙测试集）、runs/margin/eval_public_test（各牙位测试集）、runs/margin/eval_geometric
set -euo pipefail
cd "$(dirname "$0")/../.."
fetch() {  # fetch <仓库名> <目标目录>
  mkdir -p "$2"
  curl -sS -L --retry 5 -o "$2/src.zip" "https://codeload.github.com/intellident-ai/$1/zip/refs/heads/main"
  python - "$2" <<'PY'
import shutil, sys, zipfile
from pathlib import Path
out = Path(sys.argv[1])
zipfile.ZipFile(out / "src.zip").extractall(out / "src")
for p in next((out / "src").iterdir()).iterdir():
    shutil.move(str(p), str(out / p.name))
shutil.rmtree(out / "src"); (out / "src.zip").unlink()
PY
}
[ -d "data/prep_margin/training Data" ] || fetch teethPreparationData data/prep_margin
[ -d data/margin_public/registered-dies ] || fetch public-datasets data/margin_public
python -c "import vtk" 2>/dev/null || python -m pip install -q vtk "numpy==$(python -c 'import numpy; print(numpy.__version__)')"  # 不让 pip 顺带升级 numpy
cd ml
P=../data/prep_margin; MP=../data/margin_public
mkdir -p ../runs/margin
python -m dental_ai.margin.learned split-public --src $MP --out $MP/split --keep-in-train "$P/training Data" "$P/test Data"
python -m dental_ai.eval_margin --data-dir "$P/test Data" --workers 8 --out ../runs/margin/eval_geometric
M=../models/margin/margin.pt
RESUME=()
# 已有模型（仓库自带）时接着训练（学习率调小；只有验证集上更好才覆盖原文件）。FRESH=1 从头训练
if [ "${FRESH:-0}" != 1 ] && [ -f "$M" ]; then RESUME=(--init "$M" --lr "${LR:-3e-4}"); echo "从 $M 接着训练"; fi
python -u -m dental_ai.margin.learned train --data-dir "$P/training Data" $MP/split/train --out "$M" \
  --epochs "${EPOCHS:-300}" --batch-size "${BATCH:-8}" "${RESUME[@]}" | tee -a ../runs/margin/train.log
python -m dental_ai.margin.learned eval --model "$M" --data-dir "$P/test Data" --out ../runs/margin/eval_test
python -m dental_ai.margin.learned eval --model "$M" --data-dir $MP/split/test --out ../runs/margin/eval_public_test
echo "结果：runs/margin/eval_test/summary.json、runs/margin/eval_public_test/summary.json（学习模型）、runs/margin/eval_geometric/margin_summary.json（几何方法）"
