# 调研记录汇总

把这个项目做过的全部调研集中记录在这里：每次查了什么、来源是什么、得出什么结论、有什么没查到、后来改正过什么。
详细分析分别在：
- [architecture.md](architecture.md)：各任务的技术调研和方案对比
- [system-architecture.md](system-architecture.md)：系统架构（前端、后端、算法服务、3D 渲染与交互、部署）
- [algorithm-architecture.md](algorithm-architecture.md)：算法架构（模型分工、数据与训练全链路）
- [product-analysis.md](product-analysis.md)：竞品功能解析
- [industry-landscape.md](industry-landscape.md)：产业界、学术界格局和 AI 提效数据

## 1. 调研时间线（2026-10-30）

| # | 主题 | 方式 | 结论 | 去向 |
|---|---|---|---|---|
| 1 | 整体技术路线 | 网络搜索：牙齿分割数据集、牙冠生成、颈缘线检测 | 找到 Teeth3DS、DMC、CrownGen、颈缘线系列论文 | architecture.md |
| 2 | dentalai.ai 网站 | 直接访问失败（域名在本环境无法解析）；改为解析用户保存的两个网页 | 解析出三步流程、批处理、修复体类别、牙冠设计参数默认值、技术栈 | product-analysis.md |
| 3 | 用户提供的 12 个链接 | GitHub 仓库直接读取；arXiv、MICCAI、Semantic Scholar、微信文章无法访问 | 读了 TGNet、DilatedToothSegNet、PointNet notebook、ToothCraft、DMC 仓库 | architecture.md |
| 4 | 3Shape、先临三维 | 网络搜索 | 3Shape Automate 的定位和价格；先临三维公开数据集；当时没找到先临 AI 设计的细节 | product-analysis.md |
| 5 | 用户离线下载的 7 个页面 | 解析 zip 中的 HTML | 3DTeethSAM、TSegAgent、ToothLDNet、DMC 原文摘要、ToothCraft 原文摘要、ToothCR | architecture.md |
| 6 | 产业界、学术界、提效数据 | 网络搜索（中英文） | 国内外厂商、研究团队、多篇 2026 年综述和荟萃分析的设计时间数据；先临 AI Design（2025-09 发布） | industry-landscape.md |
| 7 | TGNet 源码 | 克隆仓库，读推理代码 | 发现几处 README 没写清楚的接口细节，见第 4 节 | services/tgnet |

## 2. 资料总表

### 2.1 数据集

| 名称 | 规模 / 内容 | 链接 | 许可 | 状态 |
|---|---|---|---|---|
| Teeth3DS / 3DTeethSeg'22 | 1800 个口扫，900 名患者，逐顶点 FDI 标注；扫描仪：Primescan、Trios3、iTero | [GitHub](https://github.com/abenhamadou/3DTeethSeg22_challenge) | 待核对 | 未下载 |
| Teeth3DS+ / 3DTeethLand'24 | 增加牙齿关键点标注，是首个公开的 3D 牙齿关键点数据集 | [arXiv 2210.06094](https://arxiv.org/abs/2210.06094)、[arXiv 2512.08323](https://arxiv.org/html/2512.08323v1) | 待核对 | 未下载 |
| 先临三维牙齿分割数据集 | IEEE DataPort，DOI 10.21227/txce-st96 | [链接](https://ieee-dataport.org/documents/shining3d-tooth-segmentation-dataset) | 待核对 | 未下载 |
| ODD | ToothCraft 训练用的完整牙弓数据 | 见 ToothCraft 仓库 | 待核对 | 未下载 |

### 2.2 分割

| 名称 | 会议 / 年份 | 方法 | 结果 | 代码 / 权重 / 许可 | 来源 |
|---|---|---|---|---|---|
| TGNet | 3DTeethSeg'22 第一名 | 两阶段：FPS 采样粗分 + 边界感知采样精分 | 挑战赛第一 | [有代码](https://github.com/limhoyeon/ToothGroupNetwork)，有权重（Google Drive），**无 LICENSE** | 读了仓库和源码 |
| DilatedToothSegNet | J Digit Imaging 2024 | DGCNN + 扩张边卷积 | — | [有代码](https://github.com/LucasKre/dilated_tooth_seg_net)，无权重，无 LICENSE | 读了仓库 |
| 3D_teeth_segmentation | — | 原版 PointNet notebook，私有数据 | — | [MIT](https://github.com/MohamedAlaouiMhamdi/3D_teeth_segmentation) | 读了仓库 |
| 3DTeethSAM | AAAI 2026 | 多视图渲染 + 冻结 SAM2 + 三个轻量模块 + DGAP | Teeth3DS T-mIoU 91.90%，第三磨牙 83.3% | 论文 CC BY 4.0；**代码是否开源未确认** | arXiv 摘要页 + 微信文章 |
| TSegAgent | MICCAI 2026 | SAM3 + 视觉语言模型零样本，几何先验纠错 | Teeth3DS mIoU 93.37%；私有集 82.10%，整副牙列全对率 51.96% | 未确认 | 微信文章 |
| CrossTooth | — | 边界保持的 3D 牙模分割 | TSegAgent 复现的结果偏低 | — | 搜索摘要 |

### 2.3 关键点

| 名称 | 会议 | 方法 | 代码 | 来源 |
|---|---|---|---|---|
| ToothLDNet | MICCAI 2025（#3637） | 模糊牙齿区域 + 多热图 + 二分图匹配损失，不依赖分割 | [sikingbo/ToothLDNet](https://github.com/sikingbo/ToothLDNet) | MICCAI 页面 |

### 2.4 颈缘线

| 名称 | 年份 | 方法 | 来源 |
|---|---|---|---|
| Alsheghri 等，网格分割 | 2025–2026 | 网格二分类 + 交界处样条拟合，公开了数据 | [arXiv 2507.22859](https://arxiv.org/pdf/2507.22859) |
| Adaptive Point Learning | 2024 | 把颈缘线当作点云生成，带不确定性估计 | [PDF](https://publications.polymtl.ca/59628/1/2024_Alsheghri_Adaptive_Point_Learning_Uncertainty_Quantification.pdf) |
| MADCrowner（CrownSegger） | MedIA 2026 | 颈缘线分割网络 | Semantic Scholar 引用列表 |

### 2.5 牙冠 / 缺失牙生成

| 名称 | 会议 / 年份 | 方法 | 代码 / 许可 | 来源 |
|---|---|---|---|---|
| DMC | MICCAI 2023（arXiv 2501.04914） | 上下文：预备体 + 2 颗邻牙 + 3 颗对颌牙；Transformer 点云补全 + 可微泊松重建；Chamfer 0.062 | [有代码](https://github.com/Golriz-code/DMC)，无权重，无 LICENSE | 仓库 + arXiv 摘要页 |
| ToothCraft | VISAPP 2026 / CVPR GenRecon3D（arXiv 2603.26588） | 以局部牙弓为条件的 SDF 扩散；合成残缺牙训练；IoU 81.8% | [有代码](https://github.com/ikarus1211/VISAPP_ToothCraft)，[HF 权重](https://huggingface.co/DejvaX/ToothCraft)，代码 AGPL-3.0，论文 CC BY-NC-ND | 仓库 + arXiv 摘要页 |
| MADCrowner | MedIA 2026 | 模板变形（CrownDeformR）+ 颈缘线分割 | — | Semantic Scholar 引用列表 |
| DCrownFormer+ | MedIA 2025 | 预备体 + 对颌 → 网格生成和细化，改善咬合面细节 | — | Semantic Scholar 引用列表 |
| VBCD | MICCAI 2025 | 体素粗到细 + FDI 牙位提示 | — | [arXiv 2507.17205](https://arxiv.org/pdf/2507.17205) |
| CrownGen | arXiv 2025-12 | 牙齿级点云扩散，可同时生成多颗缺失牙 | — | [arXiv 2512.21890](https://arxiv.org/pdf/2512.21890v1) |
| ToothCR | PAKDD 2022 | 两阶段：补全 + 重建 | 原文在 Springer，未读 | Semantic Scholar 页面（无摘要） |

### 2.6 竞品

| 产品 | 形态 | 关键信息 | 来源 |
|---|---|---|---|
| dentalai.ai | Web 批处理 | 分割 → 颈缘线 → AI 牙冠；每例上下颌 2 个 STL，每批最多 100 例；8 种修复体类别；牙冠参数默认值；Next.js + MUI + three.js + Stripe | 用户保存的网页 |
| 3Shape Automate | 云端 AI 设计 | 单冠 2.99 美元 / 90 秒；宣称 500 万+ 病例、94% 接受率（厂商数据） | 搜索 |
| exocad AI Design | 桌面 CAD 内置 | 2026 年一项研究中多数指标优于 Automate | 搜索 |
| Dentbird | 浏览器端 | 单病例报告约 80% 形态直接沿用 AI 结果 | 搜索 |
| 先临三维 AI Design | 齿科云平台 | 2025-09 发布；宣传单冠约 60 秒（厂商数据） | 搜索 |
| 菲森 | 全流程方案 | 宣传单冠 1.5 分钟（厂商数据） | 搜索 |

### 2.7 AI 提效数据

| 来源 | 结论 |
|---|---|
| 系统综述（PMC 2026） | 设计时间减少约 30–50% |
| GAN 牙冠综述 | 减少 60% 以上 |
| Dentbird vs 3Shape CAD | 99 分钟 vs 397 分钟（约 75%） |
| 牙支持式冠研究 | 4.5 分钟 vs 6.2 分钟 |
| Journal of Dentistry 2026 | 每颗节省 9–18 分钟，但边缘密合和咬合接触两种流程都有问题 |
| Archives of Health Investigation 2026 | 合并后无显著差异，异质性很高 |
| 综合估计（我们的推算） | 单冠设计时间减少 50–70%，技工产能提高 2–3 倍，同等产量少用 30–50% 设计人力 |

## 3. 未完成和待核对

| 项 | 原因 | 怎么补 |
|---|---|---|
| 3DTeethSAM 的代码 / 权重是否公开 | 页面里没有写 | 去 GitHub 搜索作者和论文名 |
| Teeth3DS、先临数据集的许可 | 没读到许可条款 | 下载前确认 |
| ToothCR 原文 | Springer 付费页面 | 需要时找机构访问 |
| 先临三维设计软件 AI 分割的技术细节 | 没有公开资料 | 拿同一批扫描做实测对比 |
| 各种厂商效率数字 | 都是厂商自己宣传的 | 与技工所做实测对比 |
| 市场规模 | 各家机构估计差距很大 | 只看量级 |

## 4. 改正记录

| 日期 | 原来的说法 | 改正 |
|---|---|---|
| 2026-10-30 | arXiv 2501.04914 是“随机挖牙训练”的论文 | 它就是 DMC（MICCAI 2023）的 arXiv 版本 |
| 2026-10-30 | TGNet 推理用 `--model_name tgnet_fps`（照抄 README） | 源码里推理只认 `tgnet`，README 写错了 |
| 2026-10-30 | TGNet 适配器直接传 `.h5` 路径 | `start_inference.py` 会自动加 `.h5`；现在的服务直接调用推理管线，不经过这个脚本 |
| 2026-10-30 | TGNet 输出标签数 = 输入顶点数 | TGNet 内部会先合并重复顶点，数量可能变少；服务端已映射回原始顶点 |
| 2026-10-30 | RTX 40 系“有兼容问题”（README 原话） | 原因是 pointops 写死了只编译 sm_61 / sm_75；Dockerfile 改成可配置的架构列表 |
| 2026-10-30 | 3DTeethSAM 公开集 mIoU 92.03 | 这是 TSegAgent 作者复现的数字；3DTeethSAM 论文自己报告的是 91.90 |

## 5. TGNet 源码要点（2026-10-30，commit d7e16a4）

- 推理入口：`make_inference_pipeline("tgnet", [fps.h5, bdl.h5])`，返回的管线输入 OBJ 路径，输出 `{"sem": ..., "ins": ...}`；
- 坐标要求：Y 轴指向后方，上下颌 Z 轴方向相同；内部按 Y 轴范围归一化，所以**朝向不对会直接影响结果**；
- 读取：trimesh `process=False` 读 OBJ，然后用 open3d `remove_duplicated_vertices` 去重；
- 顶点少于 24000 时会先细分一次；FPS 采样 24000 个点；
- 编号：模型输出 1–16，映射到 11–18 / 21–28；下颌再加 20 得到 31–48；
- 依赖：`.cuda()` 写死，**必须有 NVIDIA GPU**；pointops 是自定义 CUDA 算子。
