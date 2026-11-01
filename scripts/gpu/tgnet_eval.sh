#!/bin/bash
# 安装 TGNet（不用 Docker），下载作者的权重，起分割服务，在挑战赛私有测试集上评估。
# 需要 .env 里的 GOOGLE_API_KEY（下载 TGNet 作者 Google Drive 里的权重）。TORCH_CUDA_ARCH_LIST 默认 8.9（RTX 4090）。
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
set -a; [ -f .env ] && . ./.env; set +a
TG=$ROOT/third_party/ToothGroupNetwork
if [ ! -d "$TG" ]; then
  git clone https://github.com/limhoyeon/ToothGroupNetwork.git "$TG"
  git -C "$TG" checkout d7e16a4b8fca811975c3a4c7c1429bb0a51b3083
fi
# 独立的虚拟环境（继承系统里带 CUDA 的 PyTorch），避免 TGNet 的固定版本依赖影响主环境
[ -d "$ROOT/.venv-tgnet" ] || python -m venv --system-site-packages "$ROOT/.venv-tgnet"
. "$ROOT/.venv-tgnet/bin/activate"
pip install open3d==0.17.0 trimesh scikit-learn fastapi uvicorn psutil
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
(cd "$TG/external_libs/pointops" && sed -i "s/'nvcc': \[[^]]*\]/'nvcc': ['-O2']/" setup.py && python setup.py install)
# 权重：作者 Google Drive 文件夹里的 ckpts(new).zip，解压出 tgnet_fps.h5、tgnet_bdl.h5
CK=$ROOT/models/tgnet
if [ ! -f "$CK/tgnet_fps.h5" ]; then
  python scripts/download_datasets.py fetch --name tgnet_ckpts --src gdrive://15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby --out "$ROOT/data/tgnet_ckpts"
  mkdir -p "$CK"
  find "$ROOT/data/tgnet_ckpts" -name "tgnet_fps.h5" -exec cp {} "$CK/" \;
  find "$ROOT/data/tgnet_ckpts" -name "tgnet_bdl.h5" -exec cp {} "$CK/" \;
fi
# 起服务（后台），评估完关掉
mkdir -p "$ROOT/runs"
cd "$ROOT/services/tgnet"
TGNET_DIR="$TG" TGNET_CKPT_FPS="$CK/tgnet_fps.h5" TGNET_CKPT_BDL="$CK/tgnet_bdl.h5" uvicorn server:app --port 8001 > "$ROOT/runs/tgnet_server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER' EXIT
for i in $(seq 1 60); do curl -s localhost:8001/health | grep -q '"ok":true' && break; sleep 5; done
deactivate
cd "$ROOT/ml"
mkdir -p ../runs/tgnet
for d in fair_test challenge_test100 poseidon3d/test-complex; do
  name=$(basename $d)
  python -m dental_ai.evaluation --data-dir ../data/$d --gt-dir ../data/$d --segmenter-url http://localhost:8001 --orient --out ../runs/tgnet/eval_$name | tail -14
  python -m dental_ai.evaluation --data-dir ../data/$d --gt-dir ../data/$d --segmenter-url http://localhost:8001 --orient --fix-numbering --out ../runs/tgnet/eval_${name}_fix | tail -14
done
echo "结果：runs/tgnet/eval_*/summary.json"
