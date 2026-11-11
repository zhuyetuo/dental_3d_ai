#!/bin/bash
# 训练学习式颈缘线检测模型并评估，几何方法（margin/detect.py）的评估作对照。用法：bash scripts/gpu/train_margin.sh
# 数据：teethPreparationData（52 个切牙代型，MIT），自动下载到 data/prep_margin/。可用环境变量调整：EPOCHS（默认 200）。
# 输出：models/margin/margin.pt、runs/margin/eval_test/summary.json、runs/margin/eval_geometric/margin_summary.json
set -euo pipefail
cd "$(dirname "$0")/../.."
if [ ! -d "data/prep_margin/training Data" ]; then
  # github.com 连不上时 codeload 往往还能用，所以直接下 zip
  mkdir -p data/prep_margin
  curl -sS -L --retry 5 -o data/prep_margin/src.zip https://codeload.github.com/intellident-ai/teethPreparationData/zip/refs/heads/main
  python - <<'PY'
import shutil, zipfile
from pathlib import Path
out = Path("data/prep_margin")
zipfile.ZipFile(out / "src.zip").extractall(out / "src")
for p in next((out / "src").iterdir()).iterdir():
    shutil.move(str(p), str(out / p.name))
shutil.rmtree(out / "src"); (out / "src.zip").unlink()
PY
fi
python -c "import vtk" 2>/dev/null || python -m pip install -q vtk
cd ml
mkdir -p ../runs/margin
python -m dental_ai.eval_margin --data-dir "../data/prep_margin/test Data" --workers 8 --out ../runs/margin/eval_geometric
M=../models/margin/margin.pt
RESUME=()
# 已有模型（仓库自带）时接着训练（学习率调小；只有验证集上更好才覆盖原文件）。FRESH=1 从头训练
if [ "${FRESH:-0}" != 1 ] && [ -f "$M" ]; then RESUME=(--init "$M" --lr "${LR:-3e-4}"); echo "从 $M 接着训练"; fi
python -u -m dental_ai.margin.learned train --data-dir "../data/prep_margin/training Data" --out "$M" --epochs "${EPOCHS:-200}" "${RESUME[@]}" | tee -a ../runs/margin/train.log
python -m dental_ai.margin.learned eval --model ../models/margin/margin.pt --data-dir "../data/prep_margin/test Data" --out ../runs/margin/eval_test
echo "结果：runs/margin/eval_test/summary.json（学习模型）、runs/margin/eval_geometric/margin_summary.json（几何方法）"
