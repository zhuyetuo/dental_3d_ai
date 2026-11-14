# 构建上下文是仓库根目录：docker build -f backend/Dockerfile .
# REGISTRY：连不上 Docker Hub 时换成镜像站（见 docker-compose.yml 开头）
ARG REGISTRY=docker.io
# BASE_IMAGE：网络慢时可以用已经装好 torch 的旧镜像当底（如 dental3d-backend:torchbase），下面装 torch 会直接跳过
ARG BASE_IMAGE=${REGISTRY}/library/python:3.12-slim
FROM ${BASE_IMAGE}

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /app

# CPU 版 PyTorch：深度分割模型（DEEP_MODEL）和生成模型（GEN_MODEL、RADIAL_MODEL）在 worker 里推理用。
# 放在复制代码之前：改代码不用重新下载 200 MB 的 torch
RUN pip install --retries 10 --timeout 60 torch --index-url https://download.pytorch.org/whl/cpu
COPY ml /src/ml
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install /src/ml -r /tmp/requirements.txt

COPY backend/app /app/app

ENV DENTAL_DATA_DIR=/data/cases
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
