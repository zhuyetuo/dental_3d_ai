#!/bin/bash
# 不用 Docker 直接启动服务（租的 GPU 服务器通常是容器，里面跑不了 Docker）。用法：bash scripts/serve.sh
# 用仓库自带的模型：models/segmentation/deep.pt（分割）、models/generation/toothgen_open.pt（缺失牙生成），有 GPU 用 GPU，没有就用 CPU。
# 后端 http://<服务器>:8000（PORT 可改）；装了 Node ≥ 18 时同时启动前端 http://<服务器>:5173（FRONTEND_PORT 可改）。
# 病例数据在 data/cases（DENTAL_DATA_DIR 可改）。处理任务在 API 进程里跑（没有 Redis / worker）。
# 先装依赖：bash scripts/gpu/setup.sh（GPU 服务器）或 pip install -e "ml[deep]" -r backend/requirements.txt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python -c "import torch, fastapi, dental_ai" 2>/dev/null || { echo "缺少依赖：先运行 bash scripts/gpu/setup.sh"; exit 1; }
export DEEP_MODEL="${DEEP_MODEL:-$ROOT/models/segmentation/deep.pt}"
export GEN_MODEL="${GEN_MODEL:-$ROOT/models/generation/toothgen_open.pt}"
export TEMPLATE_DIR="${TEMPLATE_DIR:-$ROOT/models/templates}"
export DEMO_DIR="${DEMO_DIR:-$ROOT/demo}"
export DENTAL_DATA_DIR="${DENTAL_DATA_DIR:-$ROOT/data/cases}"
unset REDIS_URL
mkdir -p "$DENTAL_DATA_DIR"
for f in "$DEEP_MODEL" "$GEN_MODEL"; do [ -f "$f" ] || echo "警告：没有 $f，对应的功能退回几何方法 / 需要上传标签文件"; done
pids=()
trap 'kill "${pids[@]}" 2>/dev/null' EXIT
(cd backend && exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}") & pids+=($!)
if command -v npm >/dev/null; then
  (cd frontend && { [ -d node_modules ] || npm ci; } && exec npm run dev -- --host 0.0.0.0 --port "${FRONTEND_PORT:-5173}") & pids+=($!)
else
  echo "没有 npm：只启动后端。前端需要 Node ≥ 18（apt-get install -y nodejs npm，或用 docker compose）"
fi
wait -n
