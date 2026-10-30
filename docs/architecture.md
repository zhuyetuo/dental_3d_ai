# 系统方案

> 竞品（dentalai.ai、3Shape、先临三维）解析和产品功能规划见 [product-analysis.md](product-analysis.md)。

目标：输入口扫模型（上下颌 STL/OBJ/PLY）→ 牙齿实例分割和 FDI 编号 → 识别缺牙 → 生成缺失牙 3D 模型 → 底面贴合牙龈 → 导出可打印 / 可导入 CAD 的封闭 STL。

## 1. 总体架构

```
浏览器 (React + Three.js)            frontend/
   │ 上传网格，查看分割、牙颈线和生成牙，下载 STL
   ▼
API 服务 (FastAPI)                    backend/app/main.py
   │ 开发：本地目录 + BackgroundTasks
   │ 生产：PostgreSQL（病例元数据）+ MinIO/S3（网格）+ Celery/Redis 任务队列
   ▼
GPU Worker                            backend/app/worker.py → ml/dental_ai/pipeline.py
   ├─ 1 分割         dental_ai/segmentation/   逐顶点 FDI 标签
   ├─ 2 后处理       segmentation/postprocess.py  去碎片、检查编号顺序
   ├─ 3 牙颈线       margin/cervical.py         每颗牙与牙龈的交界环
   ├─ 4 缺牙检测     detection/missing.py       缺隙 / 间隙关闭 / 游离端
   ├─ 5 缺失牙生成   generation/                可替换的生成器
   └─ 6 封底         margin/base_surface.py     底面贴合牙龈，输出封闭网格
```

所有模块围绕同一种数据格式：**网格 + 逐顶点 FDI 标签（0 = 牙龈）**，与 Teeth3DS / 3DTeethSeg'22 标注一致。
这样公开数据集可以直接用来训练和评估，人工修正结果也能直接回流成训练数据。

## 2. 前端

当前实现：上传、病例列表、任务状态轮询、按牙号上色的分割结果、牙颈线叠加显示、生成牙高亮和 STL 下载。

后续：
- 牙位图（11–48）点选，手动修改错误编号；
- 画笔 / 套索修正分割，修正结果保存为新的标签文件；
- 颈缘线编辑：网格上拖动样条控制点（射线求交 + 吸附到表面）；
- 生成牙的位置、旋转、缩放 Gizmo 微调后重新封底；
- 大文件上传前用 Draco 压缩。

## 3. 牙齿实例分割（任务 3）

两条技术路线：**直接处理 3D 网格 / 点云的专用网络**，以及**多视图渲染 + 2D 视觉基础模型（SAM2/SAM3）再投影回 3D**。

| 方案 | 路线 | 说明 | 环境 / 权重 / 许可 |
|---|---|---|---|
| [ToothGroupNetwork (TGNet)](https://github.com/limhoyeon/ToothGroupNetwork) | 3D 点云 | 3DTeethSeg'22 第一名；两阶段：FPS 采样粗分 + 边界感知采样精分 | PyTorch 1.7.1 + CUDA 11.0，自定义 pointops，≥11 GB 显存，RTX 40 系有兼容问题；有权重（Google Drive）；仓库无 LICENSE |
| [DilatedToothSegNet](https://github.com/LucasKre/dilated_tooth_seg_net) | 3D 点云 | DGCNN 风格 + 扩张边卷积，17 类输出 | PyTorch ≥ 2.1，CUDA ≥ 12，pointnet2_ops，有 Dockerfile；无权重；仓库无 LICENSE |
| [3D_teeth_segmentation](https://github.com/MohamedAlaouiMhamdi/3D_teeth_segmentation) | 3D 点云 | 原版 PointNet notebook，私有数据 | MIT，只适合当入门参考 |
| [3DTeethSAM (AAAI 2026, arXiv 2512.11557)](https://arxiv.org/abs/2512.11557) | 多视图 + SAM2 | 网格归一化（牙冠朝上）后从预定义视角渲染 512×512 图像 → 冻结的 SAM2 + 三个轻量模块（提示嵌入生成器：16 个查询对应最多 16 颗牙，自动产生提示；UNet 掩码细化器：修牙龈-牙冠边界；掩码分类器：输出牙号，带“缺失”背景类）+ 编码器中插入 DGAP 可变形注意力 → 多视图投票回投 3D → 图割修边界。Teeth3DS 上 T-mIoU 91.90%、OA 95.48%、B-IoU 70.05%、Dice 94.33%，比 TGNet 高 1.74 个点；第三磨牙 mIoU 从 68.2% 提到 83.3% | 浙江大学团队（周昆、郑友怡）；只训练适配器，SAM2 权重冻结，训练成本低；论文 CC BY 4.0，代码是否开源未确认 |
| TSegAgent（MICCAI 2026） | 多视图 + SAM3 + 视觉语言模型，零样本 | 不训练牙科网络：五个基础视角加 10°–30° 扰动和曲率热图渲染，SAM3 用 “tooth” 提示分实例，回投 3D 后跨视图合并；质心拟合二次牙弓曲线排序，附上定向包围盒尺寸，交给视觉语言模型分四轮：去除非牙齿 → 找中切牙 → 全牙弓编号 → 双侧 / 顺序 / 尺寸一致性纠错 | 山东大学 / 萨里大学 / 伦敦国王学院；依赖商用大模型 API，推理慢、成本高，适合作为兜底和自动纠错，不适合作为主模型 |

**跨来源泛化是关键发现**（TSegAgent 论文 Table 1，私有集为 340 个诊所模型，多为石膏模型扫描，伪影多）：

| 方法 | Teeth3DS mIoU | 私有集 mIoU | 私有集整副牙列全对率 |
|---|---|---|---|
| MeshSegNet | 76.46 | 38.69 | 39.41 |
| PTv3 | 83.61 | 36.42 | 21.76 |
| DBGANet | 83.51 | 41.59 | 22.06 |
| 3DTeethSAM | 92.03 | 81.42 | 37.06 |
| TSegAgent | 93.37 | 82.10 | 51.96 |

注：3DTeethSAM 这一行是 TSegAgent 作者复现的结果，所以公开集 mIoU 是 92.03，和 3DTeethSAM 论文自己报告的 91.90 略有出入。

3D 专用网络在公开集上分数不错，换一个数据来源 mIoU 就跌到 40% 左右；基于 2D 基础模型的方法能保持 80% 以上。
我们的产品要面对各种口扫仪和石膏模型扫描，**泛化能力比公开榜单分数更重要**。
同时，即使最好的方法，在私有集上也有约一半的模型存在编号错误，所以**前端的人工修正功能必不可少**。

**建议路线（已根据上述结果调整）**：
1. **短期**：用 TGNet 官方权重在独立 Docker 环境里跑推理（适配器 `segmentation/tgnet.py` 已写好），尽快在真实数据上跑通整条流程；
2. **主模型**：采用 3DTeethSAM 的思路，即多视图渲染 + 冻结 SAM2 + 轻量适配器。只训练少量参数，泛化好，也能复用 SAM2 的持续升级。代码若未开源，就按论文复现；
3. **编号纠错**：借鉴 TSegAgent 的几何先验，即牙弓曲线排序、双侧对称、包围盒尺寸。`segmentation/postprocess.py` 里的 `check_arch_order` 已经在做顺序和左右检查，下一步扩展成自动纠正；视觉语言模型只在规则判断不了时调用；
4. **后处理**：去除碎片（已实现）；网格图割平滑边界（待做，两篇 SAM 论文都用了）；
5. **数据增强**：做左右镜像时必须同时交换左右牙号（1x↔2x、3x↔4x），否则模型会学乱左右；
6. **评估**：除了 mIoU，还要统计“整副牙列全对率”，并且一定要在非 Teeth3DS 来源的数据上测试，比如自有数据或先临三维数据集。

数据集：[Teeth3DS / 3DTeethSeg'22](https://github.com/abenhamadou/3DTeethSeg22_challenge)，1800 个口扫，900 个病人，逐顶点 FDI 标注，扫描仪包括 Primescan、Trios3、iTero。
扩展版 Teeth3DS+ 还有关键点标注（3DTeethLand'24）。**商用前要核对数据集许可**，长期需要自有数据。

### 牙齿关键点

[ToothLDNet（MICCAI 2025，Shi / Jin / Zheng）](https://papers.miccai.org/miccai-2025/0299-Paper3637.html)：两阶段、端到端的牙齿关键点检测，**不依赖分割**。
- 先根据关键点位置定义“模糊牙齿区域”并生成掩码；
- 再结合单颗牙的局部特征和整个牙颌的全局特征，预测每颗牙的多个热图和对应概率；
- 用二分图匹配损失同时监督牙齿定位和关键点，支持每颗牙数量不定的关键点；
- 在 Teeth3DS+ 上验证。代码：[sikingbo/ToothLDNet](https://github.com/sikingbo/ToothLDNet)。

对我们的用处：牙尖、邻接点、近远中点、牙颈点这些关键点，可以直接用于：
- 生成牙的邻接距离和咬合距离控制（对应 dentalai.ai 的 Contacts / Occlusion Distance）；
- 缺隙宽度的精确测量（替代现在的“表面最小距离”）；
- 以后的正畸排牙。

## 4. 缺牙检测（任务 4 前半）

`detection/missing.py`：
- 对比完整牙列（默认不含第三磨牙），找出缺失的牙号；
- 连续缺失的几颗牙作为一段，计算两侧邻牙表面的最小距离作为缺隙宽度；
- 缺隙 ≥ 0.5 × 期望近远中径之和 → `space`（可修复）；否则 → `closed`（如正畸拔牙后间隙已关闭）；
- 只有一侧有邻牙 → `free_end`（游离端，V1 不生成）。

## 5. 缺失牙生成（任务 4 后半）

所有生成器实现 `generation/base.py` 中的 `ToothGenerator` 接口，输入上下文（本颌网格 + 标签 + 可选对颌），输出开放的牙冠外表面，再统一交给封底模块。

| 阶段 | 方法 | 状态 |
|---|---|---|
| V1 | **对侧镜像 + 缺隙拟合**（`mirror_template.py`）：取对侧同名牙，关于正中矢状面镜像，平移到缺隙中心，沿牙弓切线缩放到缺隙宽度，有对颌时下移保证咬合间隙；无对侧牙时读取模板库 `<FDI>.stl` | 已实现 |
| V2 | 统计形状模型：用 Teeth3DS 中分割出的同号牙建 PCA 模型，以邻牙接触、咬合、对称为能量项拟合系数 | 待做 |
| V3 | 深度生成，见下表 | 待做 |

V3 参考工作：

| 工作 | 方法 | 代码 / 许可 | 适用性 |
|---|---|---|---|
| [ToothCraft (VISAPP 2026 / CVPR GenRecon3D, arXiv 2603.26588)](https://github.com/ikarus1211/VISAPP_ToothCraft) | 以局部解剖上下文为条件的 SDF 扩散模型，补全缺失或破损牙冠；用增强管线从公开完整牙弓（3DS、ODD）人工生成残缺牙作为训练数据。合成测试集 IoU 81.8%、CD 0.00034，可以直接用于真实病例，与对颌的穿插很少 | 有代码和 [HF 权重](https://huggingface.co/DejvaX/ToothCraft)；代码 AGPL-3.0，商用需购买授权；论文 CC BY-NC-ND；PyTorch 2.9 | **最贴合“根据邻牙生成缺失牙”**，推荐作为 V3 首选基线 |
| [DMC (MICCAI 2023, arXiv 2501.04914)](https://github.com/Golriz-code/DMC) | 输入点云上下文 = 预备体 + 两颗邻牙 + 对颌最近的三颗牙；特征提取 → Transformer 预测缺失区域（牙冠）的特征 → 点重建头 + MLP 输出带法向的稠密点 → 可微点到网格层（SAP）重建牙冠网格。对比基线是从通用牙冠变形到目标形状的图卷积网络。平均 Chamfer 距离 0.062 | 有代码，无权重、无 LICENSE；PyTorch 1.13 | 预备体牙冠（场景 B）的参考；**上下文的选法（2 颗邻牙 + 3 颗对颌牙）可以直接借用** |
| MADCrowner（Medical Image Analysis 2026） | CrownDeformR：用多尺度口扫编码器提取解剖上下文，把初始模板变形成目标牙冠；CrownSegger：分割目标牙的颈缘线 | — | **与我们“预测模板变形”的思路一致**，同时处理颈缘线，场景 B 的重点参考 |
| DCrownFormer+（Medical Image Analysis 2025） | 根据预备体和对颌扫描生成牙冠网格，再做细化，重点改善咬合面的窝沟和牙尖细节 | — | 咬合面细节 |
| [ToothCR (PAKDD 2022)](https://doi.org/10.1007/978-3-031-05981-0_13) | 3D 牙模的两阶段方法：先补全，再重建（Zhu、Jia 等） | Semantic Scholar 页面没有摘要，原文在 Springer | 较早的工作，被引 16 次；它的引用文献里有上面的 MADCrowner、DCrownFormer+ 等后续工作 |
| [CrownGen (arXiv 2512.21890)](https://arxiv.org/pdf/2512.21890v1) | 牙齿级点云扩散，先预测缺失牙的柱形边界再生成 | — | 多颗缺失牙同时生成 |
| [VBCD (MICCAI 2025)](https://arxiv.org/pdf/2507.17205) | 体素粗到细 + FDI 牙位提示 | — | 牙位条件化思路 |

工程要点：
- 网络输出点云再泊松重建，牙尖和窝沟容易被抹平。更稳的是**预测模板网格的变形**，或者像 ToothCraft 一样用 SDF + Marching Cubes，保证输出封闭；
- 训练数据：用 Teeth3DS 随机挖牙（单颗、连续多颗），目标就是被挖掉的那颗牙，输入是周围约 2 倍牙宽的局部区域（参考 ToothCraft）；对颌取最近的三颗牙（参考 DMC）；
- 输入范围：有研究发现至少要有一个象限的输入数据，AI 才能稳定地重建牙冠，输入越多，咬合面质量越好（Broll 等，2025）；
- 评估：Chamfer 距离、与邻牙的接触间隙、与对颌的咬合间隙、技工主观评分。

## 6. 牙颈线与底面（任务 5）

分两种临床场景：

**A. 缺失牙（种植冠 / 桥体）——已实现**（`margin/base_surface.py`）
1. 生成牙冠的开放边界沿咬合方向投影到牙龈上，并向内收拢（`taper`，形成穿龈轮廓），得到虚拟颈缘线；
2. 牙冠边界与虚拟颈缘线之间生成侧壁；
3. 颈缘线内部按同心环采样，逐点投影到牙龈表面，得到贴合牙龈的底面；
4. `relief` 控制底面到牙龈的距离：> 0 留间隙，< 0 压入牙龈（卵圆形桥体）；
5. 输出封闭网格（测试中检查 `is_watertight`）。

天然牙的牙颈线由分割结果直接得到（`margin/cervical.py`：每颗牙区域的边界环 + 平滑），前端用黄线显示，后续可以用邻牙牙颈线高度约束虚拟颈缘线。

**B. 预备体牙冠——待做**
1. 颈缘线检测：网格二分类（线上 / 线下）后在交界处拟合样条（[Alsheghri 2025](https://arxiv.org/pdf/2507.22859)），或直接生成颈缘线点云（[Adaptive Point Learning 2024](https://publications.polymtl.ca/59628/1/2024_Alsheghri_Adaptive_Point_Learning_Uncertainty_Quantification.pdf)）；
2. 冠内表面 = 预备体表面外扩粘接间隙（50–100 µm），在颈缘线处与冠外表面闭合。

## 7. 开发路线

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 项目骨架；前后端打通；合成数据端到端：分割标签 → 缺牙检测 → 镜像生成 → 封底 → 前端显示 | ✅ 本次完成 |
| M1 | 下载 Teeth3DS；TGNet Docker 推理接入 `TGNetSegmenter`；上下颌成对上传；在真实口扫上验证检测、生成和封底 | 下一步 |
| M2 | 训练自己的分割模型（PyTorch 2.x）；前端牙号修正和画笔修正；**预备体颈缘线检测和编辑（场景 B）** | |
| M3 | **预备体牙冠**（粘接间隙、最小厚度、去倒凹）；批量处理；模板库和 V2 统计形状模型；邻接和咬合优化 | |
| M4 | V3 深度生成（ToothCraft 基线 → 自研）；桥、嵌体、内冠；订阅和计费 | |
| 持续 | 医生修正数据回流；数据合规（加密、匿名化） | |

## 8. 风险

- **数据**：公开数据几乎没有“技工设计好的牙冠”作为生成任务的标准答案，和技工所合作拿 CAD 病例是生成质量的关键；
- **许可**：TGNet、DMC 无 LICENSE，ToothCraft 为 AGPL-3.0，Teeth3DS 需核对许可。商用产品要么取得授权，要么在自有数据上自研模型；
- **合规**：作为医疗器械软件上市需要注册（国内 NMPA 二类 / 三类），早期就要保留数据来源和版本记录。
