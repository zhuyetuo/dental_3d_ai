#!/bin/bash
# 安装依赖并检查 GPU。用法：bash scripts/gpu/setup.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
python -c "import torch; assert torch.cuda.is_available(), '没有可用的 GPU'; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))"
pip install -e "ml[dev]" -r backend/requirements.txt
cd ml && python -m pytest -q -x tests/test_io.py tests/test_baseline.py
echo "环境就绪"
