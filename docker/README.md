# 用 Docker 跑训练和评估（方便换服务器）

网页服务见仓库根目录的 `docker-compose.yml`。这里是 GPU 训练 / 评估环境：`docker-compose.gpu.yml`。

## 为什么用 Docker

- 换服务器（3090 / 4090 / 5090）只要装驱动和 NVIDIA Container Toolkit，`docker compose build` 一次，环境和这台完全一样。
- **GPU 性能和直接装一样**：容器直接用宿主机的显卡驱动，跑的是同一套 CUDA 算子，差别通常在 1% 以内。
  唯一要注意的是共享内存（PyTorch 数据加载进程用），compose 里已经设了 `ipc: host`。
- 5090（Blackwell，sm_120）必须 CUDA 12.8 以上、PyTorch 2.7 以上的 cu128 版本。镜像用的是官方 `pytorch:2.7.1-cuda12.8-cudnn9-devel`，三种卡都支持。
  宿主机原来的 `.venv`（torch 2.5.1 + cu124）在 5090 上跑不起来。

| 显卡 | 架构 | 驱动版本 |
|---|---|---|
| 3090 | sm_86 | 525 以上 |
| 4090 | sm_89 | 525 以上 |
| 5090 | sm_120 | 570 以上 |

## 换一台新服务器

1. 装 NVIDIA 驱动（`nvidia-smi` 能看到显卡）。
2. 装 Docker 和 NVIDIA Container Toolkit（要 sudo，只装一次）：

   ```bash
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
   docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi   # 能看到显卡就装好了
   ```

3. 拉代码，数据放到仓库旁边的 `dental_data/`（或设 `DENTAL_DATA=数据目录`），然后：

   ```bash
   cd dental_3d_ai
   DOCKER_REGISTRY=docker.m.daocloud.io docker compose -f docker-compose.gpu.yml build train      # 国内连不上 Docker Hub 时加前面的变量
   docker compose -f docker-compose.gpu.yml run --rm train python -c "import torch; print(torch.cuda.get_device_name(0))"
   docker compose -f docker-compose.gpu.yml run --rm train bash scripts/gpu/train_toothgen_antagonist.sh
   ```

4. TeethSplineSeg 对比 / 微调：`docker compose -f docker-compose.gpu.yml build splineseg`（会编译 FlashAttention，30–60 分钟；
   不想等就加 `FLASH=0`，会退回 PyTorch 自带的融合注意力，结果相同、慢一些）。

## 说明

- 代码不打进镜像，仓库整个挂进容器：改代码、拉新代码都不用重建镜像，只有依赖变了才重建。
- 路径和直接装环境时一样（容器里也是 `~/zyt/dental_3d_ai`、`~/zyt/dental_data`），`scripts/gpu/` 下的脚本不用改。
  脚本里的 `. .venv/bin/activate` 由 `docker/venv-shim` 接住（什么都不做，直接用镜像里的 Python）。
- 模型权重在仓库 `models/` 下，跟着代码走；训练中间结果在 `runs/`（不提交）。
