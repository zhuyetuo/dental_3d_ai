# 外部资料清单：论文、代码仓库、数据集

记录日期：2026-11-13。这页记录项目方收集的参考资料，以及每一项对本项目“有没有用、能不能用上”的判断。
判断基于各项目的主页 / README（2026-11-13 读取）；标“未说明”的是页面上没写，标“未读到”的是页面打不开。**许可一栏在使用前要再核对一次原文。**

结论先行：

- **缺失牙生成“表面偏平滑”**：现在的模板变形网络回归到平均形态，这是方法本身的上限。资料里对症的是 CrownFusion（扩散模型 + 几何图像表示 + 对颌条件），但它的代码和数据还没放出来；VBCD、DMC 代码已公开，任务相同（牙冠生成，带邻牙和对颌），可以作为下一版生成模型的起点。
- **已经用上的**：FDI 16 数据集（真牙细节库，见 results.md）、intellident-ai/public-datasets（颈缘线，见 results.md）。
- **手里已有、下一步最值得用的数据**：Bite2Text（1000 例真实口扫，报告里有医生确认的缺牙）、Bits2Bites（200 对咬合对齐的上下颌）、Cegep 备牙病例（备牙前 / 备牙后 / 对颌）。

## 一、牙冠 / 缺失牙生成

| 资料 | 做什么 | 方法 | 数据和代码 | 对本项目的价值 |
|---|---|---|---|---|
| [CrownFusion](https://crownfusion.github.io/)（MICCAI 2026，[代码仓库](https://github.com/JohanYe/CrownFusion)） | 预备体的邻牙 + 对颌点云 → 完整的 16 号牙冠 | 牙冠展开成“几何图像”（每个像素存一个表面坐标，咬合面中心分辨率更高），微调的图像 VAE 编码，扩散 / 插值 Transformer 生成，条件是邻牙和对颌的形状编码 | 11,513 例口扫 + 技师设计的牙冠；页面上的数据链接还是占位符，仓库 README 写“代码和数据待发布”。许可未说明 | **思路价值最高，现在用不了**。它针对的正是“点云方法在牙尖窝沟处误差集中”这个问题。可以先借鉴两点：咬合贴合度指标（对颌最近 10% 的点的平均距离、穿插面积）和几何图像表示。只做 16 号牙；每 8 个样本约 5 分钟（4090） |
| [VBCD](https://github.com/lullcant/VBCD)（MICCAI 2025） | 口扫上下文（邻牙 + 对颌）→ 牙冠 | 体素方法，细节 README 未说明 | 代码已公开；数据只给了可视化用的样例；预训练权重、许可未说明 | **可以直接试的基线**。CrownFusion 的对比表里它的 Chamfer 最好（0.353 mm，CrownFusion 0.412、DMC 0.433；注意和本项目的 Chamfer 口径不一定相同）。每个病例带曲率和“有没有颈缘”属性 |
| [DMC](https://github.com/Golriz-code/DMC) | 网格补全式的牙冠设计 | README 内容未读到 | 代码已公开（3 次提交）；数据、权重、许可未说明 | 同任务的另一个基线，资料最少，优先级低于 VBCD |
| [VF-Net](https://github.com/JohanYe/VF-Net)（arXiv 2307.10895） | 16 号牙点云的变分自编码器（带流模型先验，可采样、插值） | 点云 VAE | 代码和权重已公开；数据就是 **FDI 16 数据集**（本项目已下载，`get_data_fdi16.sh`） | FDI 16 的出处。它不是“看邻牙生成牙冠”的模型，但可以当 16 / 26 的形状先验；CrownFusion 指出它的误差集中在牙尖窝沟 |
| [MDPI Processes 13(7):1968](https://www.mdpi.com/2227-9717/13/7/1968) | 未读到（网站返回 403） | — | — | 待人工查看 |

## 二、点云补全、自监督预训练

| 资料 | 做什么 | 数据和代码 | 对本项目的价值 |
|---|---|---|---|
| [PoinTr / AdaPoinTr](https://github.com/yuxumin/PoinTr)（ICCV 2021） | 残缺点云 → 完整点云，Transformer 编码-解码 | 代码 MIT；有 ShapeNet、PCN、KITTI 上的预训练权重（不是牙科数据） | 通用补全基线。输出是点云不是网格，细节同样会被平滑，**解决不了“偏平滑”**；可以作为“缺牙区补全”的对照实验 |
| [Memory-Guided Point Cloud Completion for Dental Reconstruction](https://github.com/flyingfish118/Memory-Guided-Point-Cloud-Completion-for-Dental-Reconstruction) | 牙科点云补全，基于 PoinTr 加记忆模块 | 代码 MIT；配置里用 Teeth3DS；方法细节、指标、权重 README 未说明 | “记忆模块”和本项目的真牙细节库思路相近（从存下来的真实形状里取东西）。没有权重和指标，只能当代码参考 |
| [MeshMAE](https://github.com/liang3588/MeshMAE)（ECCV 2022） | 三角网格的掩码自编码预训练 | 代码和权重已公开（ShapeNet 预训练，网格要重划分成 500 个面） | 和本项目的点云主干不兼容，500 个面对牙科细节太粗。暂不用 |
| [Teeth-Segmentation（Alsheghri）](https://github.com/Alsheghri/Teeth-Segmentation)（SPIE 2022） | 半监督牙齿分割：有标注数据的监督损失 + 无标注数据的自监督损失 | 代码“仍在整理”；带预训练模型；数据在 Google Drive（`selfsupervisedLearningDataCegep` 就是其中的无标注部分）；许可未说明 | 方法上本项目的分割已经更强（DGCNN + 注意力）；**数据有用**，见第四节 |

## 三、正畸排牙、咬合

| 资料 | 做什么 | 数据和代码 | 对本项目的价值 |
|---|---|---|---|
| [TeethGenerator](https://github.com/lcshhh/teeth_generator)（arXiv 2507.04685） | 生成成对的正畸前 / 后牙列（给排牙网络造训练数据）：每颗牙 128 个点的 VQ-VAE + 扩散，再用 Transformer 预测每颗牙的位姿 | 代码在仓库里，README 写“将发布”；数据受数据使用协议约束 | 每颗牙只有 128 个点，做不了牙冠细节。碰撞损失（上下颌不穿插）的写法可以借鉴 |
| [TADPM](https://github.com/lcshhh/TADPM)（CAGD 2024） | 正畸前的牙 → 每颗牙的刚体变换（排牙），扩散模型 | 代码已公开，依赖 MeshMAE；数据同上 | 排牙不在本项目范围。它的预处理脚本（重划分、配准）可参考 |
| [正畸前后牙列数据集](https://zenodo.org/records/11392406)（Zenodo 11392406） | 435 位患者、1060 对治疗前后模型，带牙齿分割、牙位、牙冠关键点 | 需要签数据使用协议后邮件申请，仅限学术；元数据写 CC0 但以协议为准；总量 6.7 TB | **分割训练数据的最大增量**（拥挤、深覆𬌗等困难牙列正是现在的弱项）。要走申请流程，体量很大 |
| [Tooth-wise Factorized Policy Learning](https://github.com/panankk/Tooth-wise-Factorized-Policy-Learning-for-3D-Orthodontic-Trajectory-Generation) | 分步的正畸移动轨迹生成 | 只有代码；8000 例临床数据因隐私不公开；权重、许可未说明 | 不在本项目范围，记录备查 |
| [TSegAgent](https://arxiv.org/html/2603.19684v1)（arXiv 2603.19684） | 不训练的牙齿分割：多视角渲染 + SAM3 + 大模型给牙编号 | 匿名代码链接 | Teeth3DS 上 mIoU 93.4、全部编号正确 87%，和本项目的专用模型同一水平但要调用大模型；“编号交给大模型复核”可以作为修正邻牙编号错位的思路 |
| [Bits2Bites](https://ditto.ing.unimore.it/bits2bites/)（[代码](https://github.com/AImageLab-zip/Bits2Bites)，MICCAI 2025 ODIN） | 上下颌口扫 → 咬合分类（左右矢状关系、垂直、横向、中线） | 数据要注册，CC BY-NC-SA 4.0；代码是 Pointcept 的分支（PTv3），参考成绩 macro-F1 约 0.63 | 数据见第四节。**注意官方更正**：V01 的 `Annotations.csv` 里 “Left Class” 和 “Right Class” 两列标反了，用的时候要对调 |
| [Bite2Text](https://ditto.ing.unimore.it/bite2text/)（ECCV 2026）和[一个参赛方案](https://github.com/Ali5hadman/Bite2Text-odin2026) | 上下颌口扫 + 口内照片 → 医生风格的咬合报告 | 数据要注册，CC BY-NC-SA 4.0。参赛方案代码 AGPL-3.0，带分割和照片模型的权重；流程是 分割 → 牙冠关键点（CLIK）→ 25 个咬合测量 → 15 项判断 → 套句子，推理时不用大模型 | 数据见第四节。参赛方案的“关键点 → 咬合测量”一段，可以给本项目的生成牙冠加咬合检查；AGPL 代码不能直接并入 |

## 四、已经放在训练机上的数据（`~/zyt/dental_data/`）

| 文件 | 内容 | 许可 | 状态 |
|---|---|---|---|
| `Teeth3DS.zip` | 1800 个带逐顶点牙位标注的口扫 | CC BY-NC-ND 4.0 | 分割、生成的主训练数据 |
| `poseidon3d.zip` | 190 个口扫，含拥挤、缺牙的困难病例 | CC BY 4.0 | 分割训练 + 困难病例测试 |
| `teethPreparationData-main.zip` | 52 个切牙代型，标注颈缘线以上 / 以下 | MIT | 颈缘线 |
| `public-datasets-main.zip` | 78 个代型（前牙到磨牙）+ 颈缘线点云 | 仅限研究和教育 | **已用**：颈缘线各牙位误差 0.085 → 0.038 mm |
| `fdi16_dataset.zip` | 7762 颗 16 号牙的牙冠网格 | CC BY-NC-SA | **已用**：真牙细节库（可选功能） |
| `Bits2Bites_v01.zip` | 200 位患者，上下颌 STL 各一个 + 咬合分类 | CC BY-NC-SA 4.0 | 未用。官方说明上下颌已在咬合位对齐（没有打开文件核对）；没有逐牙标注 |
| `Bite2Text.zip` | 约 1000 位患者：上下颌 STL、口内照片、医生的咬合报告（意大利语原文 + 英语翻译） | CC BY-NC-SA 4.0 | **已用**：真实缺牙测试（`eval_real_missing.py`），286 颗医生确认缺失的恒牙只检出 41%，见 results.md |
| `selfsupervisedLearningDataCegep…zip` | 24 个病例、50 个 STL：备牙前、备牙后、对颌扫描，没有标注 | 未说明 | 未用。可以看颈缘线模型在整颌备牙口扫上的实际效果（没有标准答案，只能目视） |

## 五、建议的下一步（按性价比）

1. **（已做，2026-11-13）用 Bite2Text 做真实缺牙测试**：检出率只有 41%，主要败在缺隙已闭合的牙列上。由此引出新的第一优先级：**在分割的训练数据里模拟“缺牙且间隙闭合”**（去掉一颗牙后把远中的牙整体前移），并用这 286 颗真实缺牙作为固定的测试集。
2. **生成加入对颌条件**：Bits2Bites / Bite2Text 的上下颌是咬合对齐的，用本项目的分割模型自动标注后，就有了“带对颌的缺牙生成”训练数据；同时引入 CrownFusion 的咬合贴合度和穿插指标。现在有对颌时只做整体下沉。
3. **换生成模型**：以 VBCD 的公开代码为起点，或等 CrownFusion 发布。FDI 16 可以用来预训练牙冠形状的编码器（VF-Net 的做法）。这是解决“偏平滑”的正路，工作量最大。
4. **申请 Zenodo 11392406**：1060 对正畸牙列带分割标注，针对困难牙列分割；需要签协议。
5. **颈缘线接入流程**：用 Cegep 的备牙口扫先目视检查，再决定怎么从整颌口扫里裁出预备体。

## 数据管理的注意事项

- 同一批病例会出现在多个数据集里（如 `public-datasets` 和 `teethPreparationData` 有 2 个病例重复，本项目按患者编号把它们留在训练集）。新数据进来先按患者去重，再划分测试集。
- 由别的数据集派生的数据（如 MT3D 是在 Teeth3DS 上人工构造的缺牙）不能算新的临床病例。
- 除 Poseidon3D 和 teethPreparationData 外，上面的数据都限非商业或研究用途；用它们训练出的模型同样不能商用。
