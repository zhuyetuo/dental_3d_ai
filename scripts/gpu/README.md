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
| 4 | `bash scripts/gpu/tgnet_eval.sh` | 安装 TGNet（3DTeethSeg'22 第一名），在公平测试集、挑战赛私有测试集、Poseidon3D 上评估，作为深度模型的参照 | 安装 10–20 分钟 + 评估约 40 分钟 |
| 5 | `bash scripts/gpu/splineseg_eval.sh` | TeethSplineSeg（2026，PTv3 + B 样条，论文 T-mIoU 92.62%）：独立 Python 3.11 环境 + 作者权重，在公平测试集上评估 | 安装 10–30 分钟（FlashAttention 没有预编译包时更久）+ 评估几分钟 |
| 6 | `bash scripts/gpu/crosstooth_eval.sh` | CrossTooth（CVPR 2025）点云分支：作者仓库自带权重，在公平测试集上评估 | 安装约 10 分钟 + 评估约 10 分钟 |

**公平测试集**（`data/fair_test`，174 个扫描）：Teeth3DS 官方划分和 3DTeethSeg'22 挑战赛划分大量交叉——各自 600 个测试扫描里有 426 个在对方的训练集里。
TGNet 用挑战赛划分训练，TeethSplineSeg 和我们的模型用 Teeth3DS 划分训练，只有两个测试集的交集对所有模型都是没见过的，各模型的对比以它为准。
CrossTooth 没说用哪套划分训练，它的分数要打折看。

外部模型的注意事项：
- TeethSplineSeg 权重在 Release “checkpoint” 里，脚本按 `best.ckpt` 下载（文件名是推断的），下载失败会提示手动下载的位置；
- CrossTooth 作者只公开了点云分支，网格简化用的工具也没公开（这里用普通二次误差简化代替），分数可能低于论文；
- 这几个仓库都没有 LICENSE，只用于个人学习对比。

结果在 `runs/` 下（`runs/*/summary.json` 是官方指标汇总），模型在 `models/segmentation/deep.pt`。
跑完把 `runs/` 里的汇总发给我，或者提交到仓库（汇总文件很小，模型文件 .gitignore 已排除）。

注意：
- 服务器在国内时 GitHub、Google 可能慢或连不上：用平台的“学术加速”或代理；Drive 下载走 `www.googleapis.com`；
- TGNet 的权重在作者的 Google Drive 里，脚本用同一个 `GOOGLE_API_KEY` 下载；
- 包时长的实例到期前，把 `models/segmentation/deep.pt` 和 `runs/` 拷出来（比如上传到自己的 Drive）。
