# GPU 服务器上的一键脚本

在租的 GPU 服务器（如 RTX 4090 + PyTorch 2.1 + CUDA 11.8 镜像）上，按顺序运行。都在仓库根目录执行。

```bash
git clone https://github.com/zhuyetuo/dental_3d_ai.git && cd dental_3d_ai
```

| 步骤 | 命令 | 说明 | 大约用时 |
|---|---|---|---|
| 1 | `bash scripts/gpu/setup.sh` | 安装依赖，检查 GPU | 几分钟 |
| 2 | `bash scripts/gpu/get_data.sh` | 从 Google Drive 下载 Teeth3DS + Poseidon3D，导出训练 / 测试集。需要仓库根目录 `.env` 里有 `GOOGLE_API_KEY` 和 `DRIVE_FOLDER_ID` | 10–30 分钟 |
| 3 | `bash scripts/gpu/train_deep.sh` | 训练深度分割模型（DGCNN），在测试集上用挑战赛官方指标评估，和 CPU 基线对比 | 预处理约 30 分钟 + 训练 1–3 小时 |
| 4 | `bash scripts/gpu/tgnet_eval.sh` | 安装 TGNet（3DTeethSeg'22 第一名），在挑战赛私有测试集上评估，作为深度模型的参照 | 安装 10–20 分钟 + 评估约 30 分钟 |

结果在 `runs/` 下（`runs/*/summary.json` 是官方指标汇总），模型在 `models/segmentation/deep.pt`。
跑完把 `runs/` 里的汇总发给我，或者提交到仓库（汇总文件很小，模型文件 .gitignore 已排除）。

注意：
- 服务器在国内时 GitHub、Google 可能慢或连不上：用平台的“学术加速”或代理；Drive 下载走 `www.googleapis.com`；
- TGNet 的权重在作者的 Google Drive 里，脚本用同一个 `GOOGLE_API_KEY` 下载；
- 包时长的实例到期前，把 `models/segmentation/deep.pt` 和 `runs/` 拷出来（比如上传到自己的 Drive）。
