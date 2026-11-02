#!/bin/bash
# 安装依赖并检查 GPU。用法：bash scripts/gpu/setup.sh
# 适用于两种镜像：已经带 PyTorch 的（如 PyTorch 2.1 + CUDA 11.8），或者只装了驱动的 Ubuntu 22.04（如“预装驱动 + CUDA 12.8”）。
# 只装了驱动时，这里会装系统 Python 工具和 PyTorch 2.1.2（CUDA 12.1 版，驱动 ≥ 530 都能用）；
# TGNet、CrossTooth 的环境继承这个 PyTorch，TeethSplineSeg 另用自己的 Python 3.11 + PyTorch 2.7.1 环境。
set -euo pipefail
cd "$(dirname "$0")/../.."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

if ! command -v python >/dev/null || ! python -m pip --version >/dev/null 2>&1; then
  SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)
  $SUDO apt-get update
  $SUDO apt-get install -y python3-pip python3-venv python3-dev python-is-python3 build-essential git wget
fi
if ! python -c "import torch" 2>/dev/null; then
  python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
fi
python -c "import torch; assert torch.cuda.is_available(), '没有可用的 GPU'; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))"
# 编译 TGNet / CrossTooth 的 CUDA 算子需要 nvcc
command -v nvcc >/dev/null || [ -x /usr/local/cuda/bin/nvcc ] || echo "警告：找不到 nvcc（CUDA 工具包），tgnet_eval.sh / crosstooth_eval.sh 编译算子会失败；PATH 里加上 /usr/local/cuda/bin"
python -m pip install -e "ml[dev]" -r backend/requirements.txt
cd ml && python -m pytest -q -x tests/test_io.py tests/test_baseline.py
echo "环境就绪"
