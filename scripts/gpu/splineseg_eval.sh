#!/bin/bash
# TeethSplineSeg（arXiv 2608.17291，PTv3 + B 样条牙弓先验，论文 T-mIoU 92.62%）：安装、下载作者权重、在公平测试集上评估。
# 先跑 get_data.sh（会导出 data/fair_test：两套官方训练集都没用过的 174 个扫描）。
# 环境：独立的 Python 3.11 虚拟环境（uv 创建）+ PyTorch 2.7.1 / CUDA 12.6 + spconv + FlashAttention，不影响主环境。
# 仓库没有 LICENSE：只用于个人学习对比。
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
SS=$ROOT/third_party/TeethSplineSeg
COMMIT=238b74c26833703b281685a3f152f133ba49369c
CKPT=${SPLINESEG_CKPT:-$ROOT/models/teethsplineseg/best.ckpt}
OUT=$ROOT/runs/splineseg
[ -d "$ROOT/data/fair_test" ] || { echo "缺少 data/fair_test，先运行 bash scripts/gpu/get_data.sh"; exit 1; }

if [ ! -d "$SS" ]; then
  git clone https://github.com/blackbird2003/TeethSplineSeg.git "$SS"
  git -C "$SS" checkout "$COMMIT"
fi

# 权重：Release “checkpoint” 里作者上传的那个文件。文件名是推断的（README 用 best.ckpt），下载不到就手动下载放到 $CKPT
if [ ! -f "$CKPT" ]; then
  mkdir -p "$(dirname "$CKPT")"
  if ! wget -O "$CKPT.part" https://github.com/blackbird2003/TeethSplineSeg/releases/download/checkpoint/best.ckpt; then
    rm -f "$CKPT.part"
    echo "自动下载失败：打开 https://github.com/blackbird2003/TeethSplineSeg/releases/tag/checkpoint ，"
    echo "把 Assets 里的权重文件下载到 $CKPT （或设置 SPLINESEG_CKPT=文件路径）后重新运行"
    exit 1
  fi
  mv "$CKPT.part" "$CKPT"
fi

# 独立环境：requirements 固定了 scipy 1.16（要 Python ≥ 3.11）和 torch 2.7.1
VENV=$ROOT/.venv-splineseg
if [ ! -x "$VENV/bin/python" ]; then
  pip install -q uv
  uv venv -p 3.11 "$VENV"
fi
PY=$VENV/bin/python
if ! "$PY" -c "import torch, spconv, flash_attn, torch_scatter" 2>/dev/null; then
  uv pip install -p "$PY" pip setuptools wheel ninja packaging
  uv pip install -p "$PY" torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
  uv pip install -p "$PY" torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.7.1+cu126.html
  mkdir -p "$OUT"
  grep -vE '^(torch|torch-scatter|flash-attn)==' "$SS/requirements.txt" > "$OUT/requirements.txt"
  uv pip install -p "$PY" -r "$OUT/requirements.txt"
  # FlashAttention：setup.py 会先找对应的预编译 wheel，找不到才本地编译（很慢）
  "$PY" -m pip install flash-attn==2.8.3 --no-build-isolation
fi
"$PY" -c "import torch; assert torch.cuda.is_available(), 'PyTorch 看不到 GPU'; print('GPU:', torch.cuda.get_device_name(0))"

# 数据：链接成 Teeth3DS 原始结构，testing_*.txt = 公平测试集，training_*.txt 为空（只预处理要评估的扫描）
mkdir -p "$OUT"
LAYOUT=$ROOT/data/splineseg_root
(cd ml && python -m dental_ai.external layout --data-dir ../data/fair_test --out "$LAYOUT")
cd "$SS"
"$PY" prepare_dataset.py --source "$LAYOUT" --root "$SS/data/fair"
# 推理结束时还会用作者自己的 evaluate.py 算一遍（postprocessed_summary.txt），可以和我们的评估对照
"$PY" inference.py --ckpt "$CKPT" --devices 0 --root_folder "$SS/data/fair" --output_dir "$OUT/infer" --overwrite 2>&1 | tee "$OUT/inference.log"
cd "$ROOT/ml"

# 转成 FDI 标注，用和其它模型相同的官方指标评估（原始输出 / 加我们的编号纠错）
python -m dental_ai.external import --scheme splineseg --pred-dir "$OUT/infer/postprocessed" --data-dir ../data/fair_test --out "$OUT/pred"
python -m dental_ai.evaluation --data-dir ../data/fair_test --gt-dir ../data/fair_test --pred-dir "$OUT/pred" --out "$OUT/eval_fair" | tail -14
python -m dental_ai.evaluation --data-dir ../data/fair_test --gt-dir ../data/fair_test --pred-dir "$OUT/pred" --fix-numbering --out "$OUT/eval_fair_fix" | tail -14
echo "结果：runs/splineseg/eval_fair*/summary.json；推理耗时见 runs/splineseg/infer/timing.json"
