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

| 方案 | 说明 | 环境 | 权重 | 许可 |
|---|---|---|---|---|
| [ToothGroupNetwork (TGNet)](https://github.com/limhoyeon/ToothGroupNetwork) | 3DTeethSeg'22 第一名；两阶段：FPS 采样粗分 + 边界感知采样精分 | PyTorch 1.7.1 + CUDA 11.0，自定义 pointops，≥11 GB 显存，RTX 40 系有兼容问题 | 有（Google Drive） | 仓库无 LICENSE，商用需联系作者 |
| [DilatedToothSegNet](https://github.com/LucasKre/dilated_tooth_seg_net) | DGCNN 风格 + 扩张边卷积，17 类输出 | PyTorch ≥ 2.1，CUDA ≥ 12，pointnet2_ops，有 Dockerfile | 无，需要自己在 Teeth3DS 上训练 | 仓库无 LICENSE |
| [3D_teeth_segmentation](https://github.com/MohamedAlaouiMhamdi/3D_teeth_segmentation) | 原版 PointNet notebook，私有数据 | — | 无 | MIT，只适合当入门参考 |

**建议路线**：
1. 先用 TGNet 官方权重在独立 Docker 环境里跑推理，已写好命令行适配器 `segmentation/tgnet.py`，尽快拿到真实数据上的分割结果；
2. 同时在新环境（PyTorch 2.x）里训练 DilatedToothSegNet 或基于 Point Transformer V3 的模型，作为长期维护的主模型；
3. 后处理：去除碎片（已实现）、编号顺序检查（已实现），后续加网格图割平滑边界；
4. 数据增强做左右镜像时必须同时交换左右牙号（1x↔2x、3x↔4x），否则模型会学乱左右。

数据集：[Teeth3DS / 3DTeethSeg'22](https://github.com/abenhamadou/3DTeethSeg22_challenge)，1800 个口扫，900 个病人，逐顶点 FDI 标注，扫描仪包括 Primescan、Trios3、iTero。
扩展版 Teeth3DS+ 还有关键点标注（3DTeethLand'24）。**商用前要核对数据集许可**，长期需要自有数据。

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
| [ToothCraft (VISAPP 2026)](https://github.com/ikarus1211/VISAPP_ToothCraft) | 以周围牙弓为条件的 SDF 扩散模型，补全缺失或破损牙冠；训练数据由完整牙弓合成（AugmentPipeline） | 有代码和 [HF 权重](https://huggingface.co/DejvaX/ToothCraft)；AGPL-3.0，商用需购买授权；PyTorch 2.9 | **最贴合“根据邻牙生成缺失牙”**，推荐作为 V3 首选基线 |
| [DMC (MICCAI 2023)](https://github.com/Golriz-code/DMC) | 点云补全 Transformer + 可微泊松重建（SAP）得到网格，面向预备体牙冠 | 有代码，无权重、无 LICENSE；PyTorch 1.13 | 预备体牙冠场景参考 |
| [arXiv 2501.04914](https://arxiv.org/pdf/2501.04914) | 从完整牙弓中随机挖掉一颗牙，训练网络补回 | — | 训练数据构造思路：可以直接用 Teeth3DS 零成本造数据 |
| [CrownGen (arXiv 2512.21890)](https://arxiv.org/pdf/2512.21890v1) | 牙齿级点云扩散，先预测缺失牙的柱形边界再生成 | — | 多颗缺失牙同时生成 |
| [VBCD (MICCAI 2025)](https://arxiv.org/pdf/2507.17205) | 体素粗到细 + FDI 牙位提示 | — | 牙位条件化思路 |
| [MICCAI 2025 #3637](https://papers.miccai.org/miccai-2025/0299-Paper3637.html)、[arXiv 2512.11557](https://arxiv.org/html/2512.11557v1)、ToothCR | 用户提供的参考，当前环境无法访问原文，待补充 | — | — |

工程要点：
- 网络输出点云再泊松重建，牙尖和窝沟容易被抹平。更稳的是**预测模板网格的变形**，或者像 ToothCraft 一样用 SDF + Marching Cubes，保证输出封闭；
- 训练数据：用 Teeth3DS 随机挖牙（单颗、连续多颗），目标就是被挖掉的那颗牙，输入是周围约 2 倍牙宽的局部区域（参考 ToothCraft）；
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
