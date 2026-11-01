#!/bin/bash
# CrossTooth（CVPR 2025，北航）点云分支：安装、在公平测试集上评估。权重就在作者仓库里（models/PTv1/point_best_model.pth）。
# 先跑 get_data.sh（导出 data/fair_test）。仓库没有 LICENSE：只用于个人学习对比。
# 注意（详见 crosstooth_predict.py）：
# - 作者只公开了“只用点特征”的推理（多视图图像特征那部分没公开），简化网格用的 selective_downsample.exe 也没公开，
#   这里用普通二次误差简化代替，分数可能低于论文；
# - 作者没说用哪套划分训练（示例病例 YBSESUN6_upper 在两套官方训练集里都有），公平测试集可能有它见过的扫描。
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
CT=$ROOT/third_party/CrossTooth_CVPR2025
PT=$ROOT/third_party/point-transformer
OUT=$ROOT/runs/crosstooth
[ -d "$ROOT/data/fair_test" ] || { echo "缺少 data/fair_test，先运行 bash scripts/gpu/get_data.sh"; exit 1; }

if [ ! -d "$CT" ]; then
  git clone https://github.com/XiShuFan/CrossTooth_CVPR2025.git "$CT"
  git -C "$CT" checkout a637a9d37567d5364cad5eea6f7db91d7189a9ee
fi
# 模型代码 import models.PointTransformer.libs.pointops，仓库里没有这个目录；README 让从官方 Point Transformer 仓库装
if [ ! -d "$PT" ]; then
  git clone --depth 1 https://github.com/POSTECH-CVLab/point-transformer.git "$PT"
fi
mkdir -p "$CT/models/PointTransformer/libs"
touch "$CT/models/__init__.py" "$CT/models/PointTransformer/__init__.py" "$CT/models/PointTransformer/libs/__init__.py"
[ -e "$CT/models/PointTransformer/libs/pointops" ] || cp -r "$PT/lib/pointops" "$CT/models/PointTransformer/libs/pointops"
touch "$CT/models/PointTransformer/libs/pointops/__init__.py" "$CT/models/PointTransformer/libs/pointops/functions/__init__.py"

# 独立虚拟环境（继承系统里带 CUDA 的 PyTorch），编译 pointops
VENV=$ROOT/.venv-crosstooth
[ -d "$VENV" ] || python -m venv --system-site-packages "$VENV"
. "$VENV/bin/activate"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
if ! python -c "import pointops_cuda" 2>/dev/null; then
  pip install vedo==2023.4.4 einops trimesh scipy fast-simplification
  python -c "import torchvision" 2>/dev/null || pip install torchvision
  (cd "$CT/models/PointTransformer/libs/pointops" && python setup.py install)
fi
python -c "import torch; assert torch.cuda.is_available(), 'PyTorch 看不到 GPU'; print('GPU:', torch.cuda.get_device_name(0))"
mkdir -p "$OUT"
(cd "$CT" && python "$ROOT/scripts/gpu/crosstooth_predict.py" --repo "$CT" --data-dir "$ROOT/data/fair_test" --out "$OUT/txt") 2>&1 | tee "$OUT/inference.log"
deactivate

cd "$ROOT/ml"
python -m dental_ai.external import --scheme crosstooth --pred-dir "$OUT/txt" --data-dir ../data/fair_test --out "$OUT/pred"
python -m dental_ai.evaluation --data-dir ../data/fair_test --gt-dir ../data/fair_test --pred-dir "$OUT/pred" --out "$OUT/eval_fair" | tail -14
python -m dental_ai.evaluation --data-dir ../data/fair_test --gt-dir ../data/fair_test --pred-dir "$OUT/pred" --fix-numbering --out "$OUT/eval_fair_fix" | tail -14
echo "结果：runs/crosstooth/eval_fair*/summary.json"
echo "如果识别率接近 0 而牙齿 IoU 正常，多半是左右对应反了：把 dental_ai/external.py 里 CROSSTOOTH_* 的前后 8 个互换后重新 import"
