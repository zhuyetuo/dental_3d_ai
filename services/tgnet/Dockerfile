# TGNet（3DTeethSeg'22 第一名）分割服务，需要 NVIDIA GPU。
#
# 官方只在 pytorch 1.7.1 + CUDA 11.0 上测试过，但 CUDA 11.0 不支持 RTX 30/40 系（sm_86/sm_89）。
# 默认改用 pytorch 1.13.1 + CUDA 11.6，可以编译 sm_86，并带 PTX 让 RTX 40 系运行。
# 如需官方测试环境：--build-arg BASE_IMAGE=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel --build-arg TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0+PTX"
ARG BASE_IMAGE=pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
FROM ${BASE_IMAGE}

ARG TGNET_COMMIT=d7e16a4b8fca811975c3a4c7c1429bb0a51b3083
ARG TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/limhoyeon/ToothGroupNetwork.git /opt/ToothGroupNetwork \
    && git -C /opt/ToothGroupNetwork checkout ${TGNET_COMMIT}

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# pointops 原 setup.py 写死了 sm_61/sm_75 两种架构，改成按 TORCH_CUDA_ARCH_LIST 编译。
# 编译只需要 CUDA 工具链，不需要 GPU。
RUN cd /opt/ToothGroupNetwork/external_libs/pointops \
    && sed -i "s/'nvcc': \[[^]]*\]/'nvcc': ['-O2']/" setup.py \
    && python setup.py install

COPY server.py /app/server.py
WORKDIR /app
ENV TGNET_DIR=/opt/ToothGroupNetwork \
    TGNET_CKPT_FPS=/models/tgnet_fps.h5 \
    TGNET_CKPT_BDL=/models/tgnet_bdl.h5
EXPOSE 8001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
