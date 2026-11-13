# GPU 服务器上的一键脚本

在租的 GPU 服务器（如 RTX 4090 + PyTorch 2.1 + CUDA 11.8 镜像）上，按顺序运行。都在仓库根目录执行。

```bash
git clone https://github.com/zhuyetuo/dental_3d_ai.git && cd dental_3d_ai
```

| 步骤 | 命令 | 说明 | 大约用时 |
|---|---|---|---|
| 1 | `bash scripts/gpu/setup.sh` | 安装依赖，检查 GPU | 几分钟 |
| 2 | `bash scripts/gpu/get_data.sh` | 从 Google Drive 下载 Teeth3DS + Poseidon3D，导出训练 / 测试集。需要仓库根目录 `.env` 里有 `GOOGLE_API_KEY` 和 `DRIVE_FOLDER_ID` | 10–30 分钟 |
| 3 | `bash scripts/gpu/train_deep.sh` | 训练深度分割模型（DGCNN + 自注意力，训练数据里加模拟缺牙），在测试集上用挑战赛官方指标和缺牙基准评估 | 预处理约 1 小时（CPU 核数少时更久）+ 训练 2–3 小时 |
| 3b | `bash scripts/gpu/train_toothgen.sh` | 训练学习式缺失牙生成模型（模板变形网络），并和对侧镜像 / 模板做同口径对比 | 预处理约 5 分钟 + 训练约 1.5–2 小时 + 对比约 20–40 分钟 |
| 3c | `bash scripts/gpu/train_margin.sh` | 训练学习式颈缘线检测模型（52 个切牙代型 + 78 个各牙位代型，都自动下载），和几何方法对比 | 约 1.5 小时 |
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

显卡：**RTX 4090 最合适**（TeethSplineSeg 作者的参考硬件，配置建议按它估算）；**3090 也可以**（24 GB 显存一样，速度约 4090 的 60–70%，脚本自动按显卡编译算子）；**5090 只能跑自己的模型**（Blackwell 架构要 CUDA 12.8 + PyTorch 2.7 以上，`setup.sh` 会装；TGNet、CrossTooth、TeethSplineSeg 依赖的旧版本不支持它，要跑这三个外部模型的评估就不要选）。

配置建议（以 RTX 4090 为例）：1 卡、16 核、64 GB 内存就够；系统盘 40 GB 放不下几个虚拟环境，**把仓库放在数据盘上**
（数据 + 导出约 25 GB，三个虚拟环境约 20 GB，训练缓存和结果约 10 GB，数据盘 200 GB 足够）。只装驱动的 Ubuntu 22.04 镜像也可以，
`setup.sh` 会装 PyTorch。服务器要能访问 Google Drive（`www.googleapis.com`）、GitHub、PyTorch 和 PyPI，国内机房选境外 / 香港地域或开代理。

服务器访问不了 Google 时（国内机房常见）：Teeth3DS 和 Poseidon3D 可以直接从 OSF（`osf.io/xctdy`，`scripts/download_datasets.py teeth3ds` 能列出全部文件）
和 Zenodo（`zenodo.org/records/15608906`）下载，按 `data/drive/raw/Teeth3DS/{data_part_*.zip, Teeth3DS+/<划分目录>}`、`data/drive/raw/Poseidon3D/poseidon3d.zip` 摆放，
然后跑 `get_data.sh` 里下载之后的导出步骤。Zenodo 单连接很慢，分段并行下载（`curl -r`）快得多。

### 换服务器

代码和三个训练好的模型都在仓库里，换一台服务器（3090 / 4090 / 5090 都可以）后：

```bash
git clone https://github.com/zhuyetuo/dental_3d_ai.git && cd dental_3d_ai
bash scripts/gpu/setup.sh
```

- **启动服务**（不需要数据集）：`bash scripts/serve.sh`，后端在 8000 端口，装了 Node 时前端在 5173 端口；有 Docker 的机器用 `docker compose up --build`；
- **接着训练**：先下载数据（见下面“获取数据的两种方式”，约 10 GB，10–30 分钟），再跑 `train_deep.sh` / `train_toothgen.sh` / `train_margin.sh`，它们会从仓库自带的模型接着训练；
- 训练出更好的模型后，`git add models/... && git commit && git push` 提交回仓库，下一台服务器就能接着用。预处理缓存（`data/deep_cache`、`data/gen_cache`）不进仓库，每台服务器重新生成（共约 30 分钟）。

**获取数据的两种方式**（都不需要 API 密钥，结果一样，放在 `data/drive/raw/`）：

| 方式 | 命令 | 适用 |
|---|---|---|
| Google Drive 共享文件夹 | `bash scripts/gpu/get_data_drive.sh` | 能访问 Google 的机房（境外 / 香港）。默认用项目自己的两个共享文件夹（写在脚本里）；换别的文件夹时把链接作为两个参数，或写在 `.env` 的 `TEETH3DS_FOLDER`、`POSEIDON3D_FOLDER` 里 |
| OSF + Zenodo 官方源 | `bash scripts/gpu/get_data_osf.sh` | 任何机房，国内实测可用 |

**在 RTX 4090 上接着训练（2026-11-13）**。换到一台 4090（Ubuntu 22.04、torch 2.5.1+cu124、64 线程）后，仓库自带的分割模型先复现出和 3090 上相同的结果（综合分 96.63%、T-mIoU 86.81%），然后三个模型依次接着训练了一晚：

| 模型 | 训练量 | 之前 → 之后 |
|---|---|---|
| 分割 deep.pt | 40 + 50 轮（每轮 2.1 分钟，batch 12 占 22 GB） | 验证集点准确率 94.08% → 94.81%；公平测试集综合分 96.6% → 96.5%、T-mIoU 86.8% → 87.3%；缺牙检出 86% / 90% → 91% / 94%，完全正确 72% / 71% → 79% / 78%；Poseidon3D 综合分 87.9% → 87.3% |
| 生成 toothgen_open.pt | 60 + 100 轮（每轮 1.1 分钟，batch 64 占 22 GB） | 验证集 Chamfer 0.401 → 约 0.39 mm；测试集四个场景 0.32 / 0.33 / 0.47 / 0.47 → 0.30 / 0.32 / 0.47 / 0.46 mm |
| 颈缘线 margin.pt | 加入新数据后 300 轮 | 见上表 |

结论：数据不变时多训练，分割和生成只有很小的变化（常规口扫上的数字在波动范围内，缺牙检测有实际提升）；颈缘线的提升来自新数据。注意显存：生成模型 batch 64、分割 batch 12、颈缘线 batch 8 各自都占 20 GB 以上，24 GB 的卡上只能一个个训练。分割的预测和评估主要吃 CPU（174 个扫描约 27 分钟），训练缓存 64 线程约 12 分钟。

**真牙细节库（实验性，默认关闭）**。给学习式生成的牙冠换上真牙的牙尖窝沟（`ml/dental_ai/generation/detail.py`，效果和结论见 docs/results.md：外观更像真牙，但和真牙的距离略大约 0.02 mm）。建库（约 5 分钟，490 MB，需要 Teeth3DS 和 FDI 16 数据）：

```bash
cd ml && S=../data/drive/raw/Teeth3DS/Teeth3DS+/Teeth3DS_train_test_split
python -m dental_ai.generation.detail build --zips ../data/drive/raw/Teeth3DS --split $S/training_lower.txt $S/training_upper.txt \
  --fdi16 ../data/fdi16/train --out ../data/gen_library --workers 32
```

启用：启动服务或跑流程前 `export GEN_LIBRARY=$PWD/data/gen_library`；不设就是原来的输出。对比图：`python scripts/make_detail_figure.py`（需要 runs/gen_compare/{learned,detail3} 的评估结果）。

**FDI 16 数据集（可选，现有三个模型都不用它）**：`bash scripts/gpu/get_data_fdi16.sh` 从 Drive 下载 3Shape / DTU 的 FDI 16 数据集到 `data/fdi16/{train,val,test}`（6.7 GB，链接写在脚本里，可用参数或 `.env` 的 `FDI16_FILE` 覆盖）。内容是 7762 颗上颌右侧第一磨牙（16）的单颗牙冠网格（4844 / 1465 / 1453，单位 mm，已对齐，带点云、曲率和到边界的距离），CC BY-NC-SA 仅限非商业。只有单颗牙、没有邻牙和牙龈，所以不能训练分割模型或现在的生成模型；用途是给 16 / 26 做牙冠形状先验或细节模板库（生成的牙表面偏平滑的问题）。这个数据集只有 Drive 一个来源（官方 DTU 源在国内机房 403）：访问不了 Google 的机器上脚本会提示先在别的电脑下载再传过来。2026-11-12 测试情况：国内机房上的提示分支和解压步骤（用同结构的小样本包）跑通；Drive 直接下载的分支没有在服务器上跑过（同一条直链在本地电脑上用 curl 下载成功）。

`get_data_drive.sh` 会自动衔接两种方式：Drive 连不上、或有文件没下全（Drive 对大文件有下载次数限制）时，缺的部分从官方源补齐，所以不确定机房网络时直接用它。原来的 `get_data.sh`（Drive API + `GOOGLE_API_KEY`）仍然可用。
Drive 这条路（gdown）**还没有在能访问 Google 的服务器上实测过**。

5090 上只有外部模型的评估脚本（TGNet / TeethSplineSeg / CrossTooth）跑不了，自己的模型不受影响。**5090 上没有实测过**，装的是 CUDA 12.8 版 PyTorch。

**接着训练**：`train_deep.sh`、`train_toothgen.sh`、`train_margin.sh` 发现 `models/` 下已有对应的模型（仓库自带）时，从它接着训练（`--init`，学习率默认调小到 3e-4，用 `LR` 改），只有验证集上比原来好才覆盖原文件；`FRESH=1` 从头训练。

### 已有结果（RTX 3090，2026-11-11）

| 模型 | 数据 | 结果 |
|---|---|---|
| 深度分割（见下表） | 公平测试集 174 个扫描、缺牙基准、Poseidon3D 困难病例 | 默认模型（dgcnn_attn）：综合分 96.6%，T-mIoU 86.8%；Poseidon3D 综合分 87.9% |
| 学习式缺失牙生成（60 轮 124 分钟） | 公平测试集，模拟缺牙后跑完整流程，Chamfer 中位数（mm），对侧镜像 / 模板 → 学习模型 | 单颗缺失 0.47 → 0.32；左右同名牙都缺 0.70 → 0.33；游离端 0.62 → 0.47；两侧游离端 0.88 → 0.47 |
| 学习式颈缘线（加入 78 个各牙位代型后接着训练 300 轮，4090 上 78 分钟） | 11 个切牙代型 + 15 个各牙位代型（按患者划出的测试集） | 平均误差中位数 0.053 mm / 0.038 mm（之前只用切牙训练：0.085 / 0.085，各牙位上最差 2.17 mm → 0.46 mm）；几何方法（初始线偏 ±0.3 mm）约 0.13 mm |

**分割模型和缺牙**（2026-11-12）。只用完整牙列训练的模型，遇到缺牙的口扫会把缺牙区也认成牙（“幻影牙”），流程就检测不到缺牙。在训练数据里给每个扫描加 2 个模拟缺牙的变体（`deep cache --simulate-missing 2`，随机去掉 1–3 颗牙）后解决；换成带全局自注意力的结构（`--arch dgcnn_attn`，每个点能看到整个牙弓）进一步改善编号和困难病例。

| 模型 | 公平测试集综合分 / T-mIoU | 缺 1 颗：幻影牙 / 检出 / 完全正确 | 缺 2–3 颗：幻影牙 / 检出 / 完全正确 | Poseidon3D 困难病例综合分 / T-mIoU / 全对 |
|---|---|---|---|---|
| DGCNN，只有完整牙列（旧的 deep.pt） | 96.5% / 84.5% | 41% / 23% / 20% | 50% / 18% / 3% | 84.0% / 53.5% / 30% |
| DGCNN，加模拟缺牙接着训练 30 轮（deep_dgcnn.pt） | 96.9% / 85.0% | 0% / 80% / 70% | 0% / 87% / 66% | 86.0% / 53.8% / 30% |
| DGCNN + 自注意力，加模拟缺牙从头训练 40 轮（deep.pt，默认） | 96.6% / 86.8% | 0% / 86% / 72% | 0% / 90% / 71% | 87.9% / 59.1% / 45% |
| 同上，再加“缺牙且间隙闭合”的变体（`--simulate-closed 2`）接着训练 30 轮（现在的 deep.pt） | 96.6% / 87.4% | 0% / 92% / 79% | 0% / 93% / 76% | 89.0% / 61.5% / 40% |

最后一行在真实缺牙（Bite2Text，286 颗）上的检出率是 43%（之前 41%）；加上后处理“合并被切开的牙”（不用训练）是 55%，见 docs/results.md。

缺牙基准（`python -m dental_ai.eval_missing_seg`）：公平测试集 174 个扫描，每个模拟缺 1 颗和缺 2–3 颗；“幻影牙”= 被去掉的牙原位置仍有 20% 以上顶点被标成牙，“检出”= 缺牙检测报出了这颗牙，“完全正确”= 缺牙检测不多报也不漏报的扫描比例。幻影牙没了但检出不到 100%，剩下的错误是邻牙编号错位（比如去掉 46 后把 47 标成 46）。
注意：训练和基准用的是同一种模拟方式（把牙压平到牙龈高度），和真实的缺牙区不完全一样；Poseidon3D 是真实的困难病例，只有 20 个，其中的提升可以作为旁证。

对比里“左右同名牙都缺”“两侧游离端”两行的对照是内置的简化模板，不是用真实数据构建的模板库，对照偏弱。

注意：
- 服务器在国内时 GitHub、Google 可能慢或连不上：用平台的“学术加速”或代理；Drive 下载走 `www.googleapis.com`；
- TGNet 的权重在作者的 Google Drive 里，脚本用同一个 `GOOGLE_API_KEY` 下载；
- 包时长的实例到期前，把 `models/segmentation/deep.pt` 和 `runs/` 拷出来（比如上传到自己的 Drive）。
