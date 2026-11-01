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
| 8 | 3DTeethSeg'22 官方仓库 | 克隆仓库，读 README 和 evaluation.py | 确认 Teeth3DS 许可（CC BY-NC-ND 4.0）；官方指标的确切实现；数据在 OSF；两种划分文件 | evaluation.md、ml/dental_ai/evaluation.py |
| 9 | 公开数据集下载（2026-10-31） | 试 OSF、Hugging Face、Zenodo、Figshare（均被本环境网络策略拒绝）；在 GitHub 公开仓库里找带标注的真实扫描 | 3DTeethSeg'22 官方仓库自带 1 个带逐顶点标注的真实下颌口扫（`0EJBIPTC_lower`）和全部划分文件；TGNet、MeshSegNet 仓库没有可用的标注扫描 | scripts/download_datasets.py、real-data-test.md |
| 10 | 用户提供的 Poseidon3D 链接（2026-10-31） | 克隆 GitHub 仓库（Zenodo 访问不了） | 仓库只有 README 和示意图；数据格式说明为 tbd、没有 LICENSE 文件 | datasets.md |
| 11 | 用户上传到 Google Drive 的完整数据集（2026-10-31） | Drive API + 用户的 API 密钥下载（drive.google.com 本身被网络策略拦截） | Teeth3DS 1900 个扫描 + Teeth3DS+ 关键点 + Poseidon3D 200 个病例；Poseidon3D 格式和左右手性见 datasets.md | datasets.md、real-data-test.md 3.5 |

## 2. 资料总表

### 2.1 数据集

| 名称 | 规模 / 内容 | 链接 | 许可 | 状态 |
|---|---|---|---|---|
| Teeth3DS / 3DTeethSeg'22 | 1800 个口扫，900 名患者，逐顶点 FDI 标注；扫描仪：Primescan、Trios3、iTero；数据在 OSF（osf.io/xctdy，6 个部分） | [GitHub](https://github.com/abenhamadou/3DTeethSeg22_challenge) | **CC BY-NC-ND 4.0，仅限非商业；商用（含训练商业模型）需单独申请商业授权** | 只有官方仓库里的 1 个示例病例；完整数据待用户手动下载后上传到 GCS（见 datasets.md） |
| Teeth3DS+ / 3DTeethLand'24 | 增加牙齿关键点标注，是首个公开的 3D 牙齿关键点数据集 | [arXiv 2210.06094](https://arxiv.org/abs/2210.06094)、[arXiv 2512.08323](https://arxiv.org/html/2512.08323v1) | 待核对 | 未下载 |
| Poseidon3D | 正畸病例真实口扫，含拥挤、缺牙等困难情况；论文 Bioengineering 2024（LMVSegRNN） | [Zenodo 15608906](https://zenodo.org/records/15608906)、[GitHub](https://github.com/tiborkubik/lmvseg-poseidon3d) | 待核对（仓库无 LICENSE，README 写研究用途） | 未下载（Zenodo 在本环境访问不了）；见 datasets.md |
| 先临三维牙齿分割数据集 | IEEE DataPort，DOI 10.21227/txce-st96 | [链接](https://ieee-dataport.org/documents/shining3d-tooth-segmentation-dataset) | 待核对 | 未下载 |
| ODD | ToothCraft 训练用的完整牙弓数据 | 见 ToothCraft 仓库 | 待核对 | 未下载 |

### 2.2 分割

| 名称 | 会议 / 年份 | 方法 | 结果 | 代码 / 权重 / 许可 | 来源 |
|---|---|---|---|---|---|
| TGNet | 3DTeethSeg'22 第一名 | 两阶段：FPS 采样粗分 + 边界感知采样精分 | 挑战赛第一 | [有代码](https://github.com/limhoyeon/ToothGroupNetwork)，有权重（Google Drive），**无 LICENSE** | 读了仓库和源码 |
| DilatedToothSegNet | J Digit Imaging 2024 | DGCNN + 扩张边卷积 | — | [有代码](https://github.com/LucasKre/dilated_tooth_seg_net)，无权重，无 LICENSE | 读了仓库 |
| 3D_teeth_segmentation | — | 原版 PointNet notebook，私有数据 | — | [MIT](https://github.com/MohamedAlaouiMhamdi/3D_teeth_segmentation) | 读了仓库 |
| 3DTeethSAM | AAAI 2026 | 多视图渲染 + 冻结 SAM2 + 三个轻量模块 + DGAP | Teeth3DS T-mIoU 91.90%，第三磨牙 83.3% | 论文 CC BY 4.0；代码和权重已公开，无 LICENSE（见 7.1） | arXiv 摘要页 + 微信文章；2026-11-01 clone 核实 |
| TSegAgent | MICCAI 2026 | SAM3 + 视觉语言模型零样本，几何先验纠错 | Teeth3DS mIoU 93.37%；私有集 82.10%，整副牙列全对率 51.96%（自报；他人实测差很多，见 7.1） | 匿名仓库 | 微信文章 |
| CrossTooth | CVPR 2025 | 边界保持的 3D 牙模分割 | TSegAgent 复现的结果偏低 | [CrossTooth_CVPR2025](https://github.com/XiShuFan/CrossTooth_CVPR2025)，权重在仓库里，无 LICENSE（见 7.1） | 2026-11-01 clone 核实 |

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
| 先临数据集、ODD 的许可 | 没读到许可条款 | 下载前确认（Teeth3DS 已确认，见上表） |
| 评估指标与官方代码的直接比对 | 本环境不允许执行下载来的外部代码 | 拿到数据后用官方 evaluation.py 跑同一批预测对比 |
| 朝向归一化在真实口扫上的效果 | 只在 1 个示例病例上测过（8.7°）；本环境访问不了 OSF | 下载完整 Teeth3DS 后在测试集上跑 `--check-orientation` |
| 完整 Teeth3DS 下载脚本 | 本环境的网络策略禁止访问 osf.io、huggingface.co、zenodo.org、figshare.com | 在能访问的机器上运行 `scripts/download_datasets.py teeth3ds` |
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
| 2026-10-30 | 评估指标最初按论文文字描述实现（最近质心匹配、包围盒对角线归一化、TSA 为牙齿前景 F1） | 改为复刻官方 evaluation.py：匈牙利匹配、逐轴尺寸归一化、TSA 为二分类 micro-F1（即准确率）、TIR 分母为匹配数 |
| 2026-10-30 | Teeth3DS 许可“待核对” | CC BY-NC-ND 4.0，仅限非商业，商用需单独授权 |
| 2026-10-31 | Teeth3DS / TGNet 坐标系：上颌牙冠朝 −Z、+X 为患者左侧 | 实际数据里上下颌都是牙冠朝 +Z，上颌患者左侧在 −X（上颌模型翻过来看） |
| 2026-10-31 | Poseidon3D 与 Teeth3DS 标注约定相同 | 左右手性相反（按右手系摆正后标成左侧的牙在右侧），导出时镜像 |
| 2026-10-31 | 编号纠错“两颗牙被标成同一个号”只给警告即可 | 真实口扫上相邻牙接触，同号后连成一块，纠错会把后面的牙整体错改一位；动态规划要允许一块占两个牙位 |
| 2026-10-31 | Teeth3DS 的 OBJ 顶点都被三角形引用，用 trimesh 读入即可 | 约 7% 是孤立顶点，trimesh 会丢掉，标签就对不上；TGNet 也是 trimesh 读入，所以它的输出比顶点数少（官方示例输出正好少了这些） |
| 2026-10-30 | 3DTeethSAM 公开集 mIoU 92.03 | 这是 TSegAgent 作者复现的数字；3DTeethSAM 论文自己报告的是 91.90 |

## 5. TGNet 源码要点（2026-10-30，commit d7e16a4）

- 推理入口：`make_inference_pipeline("tgnet", [fps.h5, bdl.h5])`，返回的管线输入 OBJ 路径，输出 `{"sem": ..., "ins": ...}`；
- 坐标要求：Y 轴指向后方，上下颌 Z 轴方向相同；内部按 Y 轴范围归一化，所以**朝向不对会直接影响结果**；
- 读取：trimesh `process=False` 读 OBJ，然后用 open3d `remove_duplicated_vertices` 去重；
- 顶点少于 24000 时会先细分一次；FPS 采样 24000 个点；
- 编号：模型输出 1–16，映射到 11–18 / 21–28；下颌再加 20 得到 31–48；
- 依赖：`.cuda()` 写死，**必须有 NVIDIA GPU**；pointops 是自定义 CUDA 算子。

## 6. DMC 全文和源码要点（2026-11-01，arXiv 2501.04914v1，仓库最后更新 2023-07-11）

**方法**
- 输入：预备体所在牙弓（master）+ 对颌（antagonist）的点云，用分割模型取出预备体、2 颗邻牙、3 颗对颌牙和周围牙龈，随机采 10240 个点；
- 网络：DGCNN 分组 → PoinTr 的 Transformer 编码器/解码器 → FoldingNet 把 2D 网格折成牙冠粗点（`num_pred 1568`）→ MLP 给每个点加偏移和法向（加密）→ 可微泊松重建（SAP/DPSR，128³ 网格）→ Marching Cubes 出网格；
- 损失：Chamfer（点）+ 指示函数网格的 MSE（面）；真值指示网格离线用 DPSR 从技工设计的牙冠算好（`psr.npz`）；
- 数据：Kerenor 技工所的真实病例 388 训练 / 97 验证 / 71 测试，真值是技工设计的牙冠（文件名 `shell`），**不公开**；增强 10 倍（平移、缩放、旋转）；A100 训练 400 轮约 22 小时。

**结果**（Chamfer 和 F-score 都在归一化坐标下算：上下文点减均值、除以各轴标准差，**不是毫米**，不能和我们的 mm 指标直接比）

| 方法 | CD-L1 | CD-L2 | F-score@0.3 |
|---|---|---|---|
| PoinTr | 0.070 | 0.023 | 0.24 |
| PoinTr + 颈缘线 | 0.065 | 0.018 | 0.54 |
| PoinTr + SAP（两个网络分开） | 0.067 | 0.021 | 0.50 |
| PoinTr + 图卷积模板变形 | 1.99 | 1.51 | 0.08 |
| DMC 不加 MSE | 0.0641 | 0.015 | 0.65 |
| DMC | 0.0623 | 0.011 | 0.70 |

**源码现状**：readme 为空；数据路径写死成作者的 Windows 目录；依赖 open3d 0.9、PyTorch 1.13 和几个自编译 CUDA 扩展（仓库里还带着 py3.8 的编译产物）；没有权重、没有数据，主体代码没有 LICENSE（只有 SAP 子目录是 MIT）。**不能直接跑起来，也不能直接拿来用。**

**对我们的判断**
- 它只生成牙冠整体外形，不单独处理组织面和颈缘：128³ 网格 + Marching Cubes 的分辨率在亚毫米级，达不到颈缘密合要求（几十微米）；论文也没评估颈缘间隙、最小厚度、咬合接触。所以即使以后用学习模型，也只替换 `prep_crown.py` 的**外表面**，组织面、去倒凹、颈缘缝合、最小厚度继续用几何方法；
- 图卷积模板变形在论文里输得很惨（CD 1.99），作者的解释是“需要大量预处理去缩放和定位模板”。我们的模板路线已经做了这一步（按邻牙缺隙、牙弓、统计高度定位），所以这个对比不说明模板路线不行；
- 可借用：①上下文选法（预备体 + 2 颗邻牙 + 3 颗对颌牙）；②除了点损失再加一个“面”的损失（指示函数 / SDF），消融里 F-score 0.65 → 0.70；③输出带法向的点后用泊松重建得到封闭网格；④评估加双向 Chamfer 和 F-score（我们用 mm 为单位）；
- 训练数据：没有技工设计的牙冠时，可以在 Teeth3DS 上用 `synthetic.simulate_preparation` 造“预备体 → 原天然牙”的配对（1800 个扫描，每个多颗牙），真值是天然牙形态而不是修复体设计；Teeth3DS 没有咬合关系，对颌这一路要等有咬合数据。

## 7. 2026 年分割、牙冠生成、颈缘线调研（2026-11-01）

方法：网络搜索 + 能 clone 的仓库直接看（arXiv 在本环境访问不了，论文细节来自搜索摘要，标注为“摘要”）。
**2026-11-01 第二轮更正**：用户那边读了 arXiv 全文和 AAAI 正式版，找到了论文里的代码地址；我把这些仓库都浅克隆核实了一遍（“clone 核实”）。更正内容见 7.4。

### 7.1 分割（论文多，带代码和权重的少）

| 名称 | 时间 | 方法 | 结果 | 代码 | 核实方式 |
|---|---|---|---|---|---|
| TSegAgent（arXiv 2603.19684） | 2026-03，MICCAI 2026 | 多视图渲染 + SAM3 零样本实例分割 + 视觉语言模型编号，不需要训练 | 自报 Teeth3DS mIoU 93.37%；**B-Spline 论文实测它的开源实现：OA 62.16%、T-mIoU 31.45%，每例 372.5 秒** | 匿名仓库 anonymous.4open.science/r/TSegAgent-3FB4 | 用户读全文；不建议投入 |
| 3DTeethSAM（arXiv 2512.11557） | AAAI 2026 | 多视图渲染 + SAM2 + 投票回投到网格 | T-mIoU 91.90%，第三磨牙提升 15% 以上 | [Crisitofy/3DTeethSAM](https://github.com/Crisitofy/3DTeethSAM)，2026-03-02 更新，**无 LICENSE**；权重（SAM2 Hiera-L + 自己的模型）放在 README 里的 Google Drive 文件夹，**文件夹内容未核实** | clone 核实 |
| **B-Spline 结构学习 / TeethSplineSeg**（arXiv 2608.17291，浙大，与 3DTeethSAM 同组） | 2026-08 | PTv3 主干 + 过牙齿中心的三次 B 样条，每个点投影到曲线上得到连续的“牙弓位置”坐标，辅助分类 | 同一张表里 T-mIoU **92.62%**（TGNet 90.16%）；4090 上训练 2 小时 15 分，推理每例 0.84 秒（比 TGNet / 3DTeethSAM 训练快 9 倍多、推理快 3–4 倍） | [blackbird2003/TeethSplineSeg](https://github.com/blackbird2003/TeethSplineSeg)，2026-08-25 更新，**无 LICENSE**；README 说权重在 Release `checkpoint`（tag 存在；本环境打不开 GitHub 网页，没列出文件名）；直接读 Teeth3DS 原始 OBJ/JSON | clone 核实；**TGNet 的首选替代** |
| SOFTooth（arXiv 2512.23411） | 2025-12 | 语义增强 + 顺序感知的实例分割 | — | 未见 | 摘要 |
| CrossTooth（北航） | CVPR 2025 | 边界保持（多视图渲染特征 + PTv1）；比 TGNet 牙齿 IoU +2.3%、边界 IoU +5.7% | — | [XiShuFan/CrossTooth_CVPR2025](https://github.com/XiShuFan/CrossTooth_CVPR2025)，2025-06-21 更新，**无 LICENSE**；**权重就在仓库里**（`models/PTv1/point_best_model.pth`，27 MB） | clone 核实 |
| OrthoTwin3D | 2026-10 更新 | DGCNN 在 Teeth3DS 上的可复现训练流程（分割 → 实例 → 关键点 → 几何测量） | 验证集 mIoU 0.626（未加增强） | 有代码，**无 LICENSE**，无权重 | 读了 README |
| 分类引导的局部口扫流程（arXiv 2605.15241） | 2026-05 | 先判断全牙弓 / 局部扫描，再走不同的分割模型 | 1958 个局部扫描，DSC 0.925 | 未见 | 摘要 |

结论：2026 年的主流是“多视图渲染 + 视觉基础模型（SAM2/SAM3）”和“PTv3 + 牙弓结构先验”。公开数据上的数字都在 91–93% mIoU。
**有代码和权重的不止 TGNet**：TeethSplineSeg（Release）、CrossTooth（仓库内）、3DTeethSAM（Google Drive）都给了权重，但都没有 LICENSE（个人学习可以用，商用要找作者）。GPU 上对比的优先顺序：TeethSplineSeg（最快、分数最高、读 Teeth3DS 原格式）> CrossTooth > 3DTeethSAM；TSegAgent 不投入。B-Spline 论文的“牙弓连续坐标”和我们编号纠错里的牙弓投影（`_arch_position`）是同一个思路，
可以作为深度模型的额外输入特征。

### 7.2 牙冠 / 缺失牙生成（论文少，可用代码更少）

| 名称 | 时间 | 方法 | 代码 / 许可 | 核实方式 |
|---|---|---|---|---|
| CrownGen（arXiv 2512.21890） | 2025-12 | 牙齿级点云扩散 + 边界预测模块，一次生成多颗；496 个外部扫描 + 26 例临床研究，医生评价不劣于技工 | [baejustin/CrownGen](https://github.com/baejustin/CrownGen)，2025-12-26 更新，**无 LICENSE**，带示例数据，未见权重 | 读了 README |
| ToothForge（IPMI 2025）+ 深度谱模型（arXiv 2606.31293，2026-06，MELBA） | 2025–2026 | 谱域（拉普拉斯特征基）生成牙冠网格，不要求统一拓扑，毫秒级 | [tiborkubik/toothForge](https://github.com/tiborkubik/toothForge)，**无 LICENSE**，训练数据私有 | 读了 README |
| ToothCraft（arXiv 2603.26588） | VISAPP 2026 | 以局部牙弓为条件的 SDF 扩散，合成残缺牙训练 | [AGPL-3.0](https://github.com/ikarus1211/VISAPP_ToothCraft)，2026-10-13 仍在更新，HF 有权重 | clone |
| 分类引导的牙冠方案（arXiv 2605.15241，尼泊尔、日本企业团队） | 2026-05 | 先判断全牙弓 / 左 / 右 / 前牙局部扫描并配准到标准姿态再分割；牙冠用**检索 + 拟合**：Teeth3DS 上训练的 DGCNN 特征，按邻牙、对颌余弦相似度检索，样条对齐后用 Blender 调整邻接和咬合。作者承认检索和拟合**只有定性展示**，没和技工设计定量比 | 论文没有代码链接 | 用户读全文 |
| **MADCrowner**（arXiv 2603.04771，港中文、商汤、上海九院等） | 2026-03 | 先分割预备体、提取颈缘线（CrownSegger），再把模板牙冠变形成目标牙冠（CrownDeformR）；37.3M 参数，单次 600 ms；颈缘分割 576 例口扫（476 训练 / 100 测试），牙冠数据来自 4602 名患者（带技工设计） | [lullcant/MADCrowner](https://github.com/lullcant/MADCrowner)，2026-07-02 更新，**Apache-2.0**；带 28 颗 FDI 模板牙冠（`fdi_template/*.ply`）；**没有权重**（README 只说“在脚本里改权重路径”），数据因隐私不公开 | clone 核实；**和我们“模板库 + 定位”最接近** |
| DCrownFormer / DCrownFormer+ | MICCAI 2024 / MedIA 2025 | 预备体 + 对颌 → 网格 | 仓库只剩一句话：**技术已转让给韩国公司，不再公开代码** | clone |
| 深度自适应感知 GAN（Soft Computing 2026） | 2026-07 | GAN 生成牙冠 | 未见 | 摘要 |

结论：生成方向确实少，原因是需要“技工设计好的牙冠”做真值，这类数据都在技工所 / 企业手里，几篇有效果的工作数据都不公开，
最强的 DCrownFormer 已经商业化收回。MADCrowner 是唯一许可清楚（Apache-2.0）的，但没有权重、没有数据，要自己训练就需要技工设计数据；它的 28 颗模板牙冠可以和我们的模板库对比。2026 年有一篇明确走**检索 + 拟合**路线（不是端到端生成），和我们“模板库 + 按邻牙定位”的思路一致，
说明在没有大规模设计数据时这条路是实用的。值得跟进的是 CrownGen（有代码 + 示例数据）。

### 7.3 颈缘线（最少，但有一份公开的真实数据）

| 名称 | 时间 | 方法 | 数据 / 代码 | 核实方式 |
|---|---|---|---|---|
| Alsheghri 等，网格分割（arXiv 2507.22859；Mathematics and Computers in Simulation 239, 2026） | 2025–2026 | 预备体网格二分类，交界拟合样条；集成模型 13 例中 7 例达到 0.2 mm 的技工一致水平 | **数据公开，MIT**：[intellident-ai/teethPreparationData](https://github.com/intellident-ai/teethPreparationData)，52 个切牙代型 | clone 并评估（real-data-test.md 3.11） |
| Adaptive Point Learning（Applied Sciences 2024） | 2024 | 颈缘线作为点云生成，Chamfer + infoCD，带不确定性 | 未见 | 摘要 |
| Alsheghri 等，全牙位（Computers in Biology and Medicine 2025） | 2025 | 3D 深度学习提取颈缘线；1113 个真实预备体扫描、覆盖所有牙位；78 个测试例中 70 例在 200 μm 内、62 例在 100 μm 内 | 数据未见公开（公开的只有切牙 52 例） | 用户读摘要 |
| MADCrowner 的 CrownSegger | 2026-03 | 预备体分割 → 颈缘线；576 例口扫 | Hausdorff 距离 0.328 mm（**最大误差**，不能和我们的中位数直接比）；代码 Apache-2.0，无权重、无数据 | clone 核实 |
| 自动颈缘线检测综述（Zenodo，2026-05） | 2026 | 概述性文章，没有临床数据 | — | 摘要 |
| 专利（US 12210802、US 12236607 等） | — | 神经网络提出颈缘线 / 自动检测和修正 | — | 摘要 |

结论：颈缘线需要的标注只有技工所有。学术工作主要来自两个团队：Polytechnique Montréal + Intellident（DMC 的团队，切牙数据公开）和 MADCrowner 团队（港中文等，数据不公开）。
我们的几何检测器在这份真实数据上：初始线偏差 0.6 mm 以内时误差中位数约 0.15 mm，83–88% 的点在 0.2 mm 内（详见 real-data-test.md 3.11）。
52 个病例太少、只有切牙，训练学习模型意义不大，先作为评估集。

### 7.4 第二轮更正记录（2026-11-01）

| 第一轮说法 | 更正 |
|---|---|
| CrossTooth 仓库 clone 需要登录 | 用错了地址；正确的是 XiShuFan/CrossTooth_CVPR2025，公开，权重在仓库里 |
| 分割论文都没确认有代码和权重 | TeethSplineSeg、3DTeethSAM、CrossTooth 都有代码和权重（都没有 LICENSE） |
| TSegAgent mIoU 93.37% | 这是自报数字；B-Spline 论文实测它的开源实现只有 T-mIoU 31.45%，每例 372.5 秒 |
| MADCrowner 未见代码 | 有，lullcant/MADCrowner，Apache-2.0，无权重、无数据 |
| 颈缘线工作几乎都来自一个团队 | 漏了 MADCrowner 团队；Alsheghri 团队还有一篇 1113 例全牙位的 CBM 2025 |
| B-Spline IoU 91.90% | 91.90% 是 3DTeethSAM 的数字（搜索摘要混在一起了）；B-Spline 自己是 T-mIoU 92.62% |

**划分交叉（2026-11-01）**：Teeth3DS 划分和挑战赛划分各自 600 个测试扫描里有 426 个在对方的训练集里。
论文里“在 3DTeethSeg22 上”的数字用的是哪套划分要看清：TeethSplineSeg 用 Teeth3DS 划分，TGNet 用挑战赛划分，两者论文数字不能直接比。
统一对比用两个测试集的交集（174 个扫描，见 real-data-test.md 3.12）。
