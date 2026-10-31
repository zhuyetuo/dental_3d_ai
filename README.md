# dental_3d_ai
**口腔 3D 扫描模型（上下颌 STL）输入 → 牙齿实例分割 → 识别缺牙位置 → 自动生成缺失牙齿 3D 模型**

## 文档

| 文档 | 内容 |
|---|---|
| [docs/system-architecture.md](docs/system-architecture.md) | 系统架构：前端、后端、算法服务、3D 渲染与交互、部署 |
| [docs/algorithm-architecture.md](docs/algorithm-architecture.md) | 算法架构：模型分工、数据收集标注训练全链路 |
| [docs/architecture.md](docs/architecture.md) | 各任务的技术调研、论文和开源方案对比 |
| [docs/product-analysis.md](docs/product-analysis.md) | 竞品解析（dentalai.ai / 3Shape / 先临三维）与功能规划 |
| [docs/industry-landscape.md](docs/industry-landscape.md) | 产业界、学术界格局与 AI 提效数据 |
| [docs/real-data-test.md](docs/real-data-test.md) | 真实口扫测试记录：发现并修复的问题、缺失牙生成与真牙的对比 |
| [docs/annotation-guidelines.md](docs/annotation-guidelines.md) | 标注规范 v1.0：牙齿分割与编号（含数据接收要求、病例登记表模板、质量控制） |
| [docs/evaluation.md](docs/evaluation.md) | 分割评估（Teeth3DS、官方指标）与朝向归一化 |
| [docs/research-log.md](docs/research-log.md) | 全部调研记录汇总（时间线、资料总表、待核对、改正记录） |

## 目录

```
ml/        核心算法包 dental_ai（与 Web 无关，可单独使用）
  dental_ai/fdi.py              FDI 牙位编号工具
  dental_ai/segmentation/       分割接口、远程分割服务客户端、编号纠错、后处理
  dental_ai/detection/          缺牙检测
  dental_ai/generation/         缺失牙生成（对侧镜像 / 标准牙模板 + 缺隙拟合）、预备体牙冠生成
  dental_ai/templates/          标准牙模板库（牙齿局部坐标系、内置参数化模板、从标注数据构建）
  dental_ai/margin/             牙颈线提取、预备体颈缘线检测、贴合牙龈的底面生成
  dental_ai/preferences.py      设计参数（邻接距离、咬合距离、粘接间隙等）
  dental_ai/orientation.py      朝向归一化（分割前把任意朝向的口扫转到标准坐标系）
  dental_ai/evaluation.py       分割评估（挑战赛官方指标、TGNet 口径、论文口径）
  dental_ai/pipeline.py         端到端流程 / 命令行入口
  dental_ai/synthetic.py        合成牙弓，用于测试和联调
backend/   FastAPI 服务
services/  算法服务（每个模型一个容器）
  tgnet/                        TGNet 分割服务（需要 NVIDIA GPU）
models/    模型权重（不提交到 git，见 models/README.md）
frontend/  React + Three.js 查看器和标注编辑器（src/editor）
scripts/   工具脚本（合成数据、公开数据集下载）
data/      下载的数据集（不提交到 git）
docs/      设计文档
```

## 用 Docker 启动

前端 + 后端（没有 GPU 时，上传标签文件跳过分割）：

```bash
docker compose up --build
```

加上 TGNet 分割服务（需要 NVIDIA GPU 和 nvidia-container-toolkit，权重放法见 [models/README.md](models/README.md)）：

```bash
docker compose --profile gpu up --build
```

浏览器打开 http://localhost:8080 。

## 本地开发

需要 Python ≥ 3.10、Node ≥ 18。

```bash
pip install -e "ml[dev]" -r backend/requirements.txt
```

生成合成数据（下颌缺 36）：

```bash
python scripts/make_synthetic.py --jaw lower --missing 36 --out samples
```

命令行跑完整流程：

```bash
cd ml && python -m dental_ai.pipeline ../samples/synthetic_lower.obj --jaw lower --labels ../samples/synthetic_lower.json --out ../out
```

启动后端：

```bash
cd backend && uvicorn app.main:app --reload --port 8000
```

启动前端（另开一个终端，浏览器打开 http://localhost:5173，上传 `samples/` 下的 .obj 和 .json）：

```bash
cd frontend && npm install && npm run dev
```

下载公开数据集（Teeth3DS 许可 CC BY-NC-ND 4.0，仅限非商业研究；数据放在 `data/`，不进 git）：

```bash
python scripts/download_datasets.py teeth3ds-sample
```

`teeth3ds-sample` 是官方仓库里的 1 个真实下颌口扫（约 12 MB），用来冒烟测试；`teeth3ds-splits` 是官方划分文件；`teeth3ds` 是完整数据集（OSF，几十 GB）。

运行测试（下载了示例病例时会多跑真实数据的回归测试，否则跳过）：

```bash
cd ml && pytest -q
```

```bash
cd backend && pytest -q
```

前端单元测试（标注编辑逻辑）：

```bash
cd frontend && npm test
```

TGNet 服务的接口测试（用假模型，不需要 GPU）：

```bash
pytest -q services/tgnet/tests
```

## 标注编辑器

处理完成后，右侧就是编辑器：

| 模式 | 快捷键 | 操作 |
|---|---|---|
| 查看 | V | 左键旋转、右键平移、滚轮缩放 |
| 改牙号 | S | 在牙位图上选牙号，再点击模型上的一颗牙（或一块牙龈），整块改成该牙号 |
| 画笔 | B | 选牙号，按住左键涂抹，修正牙齿和牙龈的边界；可以调画笔半径 |
| 颈缘线 | M | 在牙位图上选牙，“设为预备体”后自动检测颈缘线；拖动青色控制点移动，Shift+点击模型添加控制点（没有颈缘线时可以逐点画），Alt+点击控制点删除；点“保存颈缘线” |

撤销 Ctrl+Z，重做 Ctrl+Shift+Z。改完点“保存并重新计算”：修正后的标签保存为新版本，并重新运行缺牙检测、生成和封底。模型的原始预测会单独保留，用作训练数据。

## 当前状态

- 分割：TGNet 已做成独立的 GPU 服务（`services/tgnet`），后端通过 `SEGMENTER_URL` 调用；上传标签文件时直接使用标签。调用前会先做朝向归一化，分割后自动做编号纠错（纠错可用 `NUMBERING_FIX=0` 关闭）；
- 真实口扫：已在 Teeth3DS 的 1 个示例病例上跑通完整流程，修复了 11 个只在真实数据上出现的问题（孤立顶点、扫描小洞、牙颈线夹点、封底非流形、两颗牙连成一块等），见 docs/real-data-test.md。完整数据集在 OSF，本开发环境访问不了，待在能访问的机器上下载后重跑；
- 带标签的数据请用 OBJ/PLY。STL 没有共享顶点，读入时会合并顶点，顶点顺序与标签文件对不上；
- 生成：外形供体优先用对侧同名牙镜像，对侧也缺（或是预备体）时用标准牙模板库；支持两侧都有邻牙的缺隙、牙弓末端（游离端，如最后一颗磨牙）和连续多颗缺牙。放置位置：沿牙弓方向按缺隙和各牙标准宽度分配，颊舌方向用对侧镜像位置或局部牙弓曲线；游离端优先用对侧同名牙的镜像位置，对侧也没有时沿牙弓局部外推。真实口扫上（模拟缺牙后与真牙比较）：对侧镜像的表面偏差 0.3–0.65 mm，内置模板约 1 mm；
- 标准牙模板库：默认用内置的参数化模板（按教科书平均尺寸程序生成的**简化形态**，只是占位）。有标注数据后用 `python -m dental_ai.templates build --data-dir <obj> --gt-dir <标注> --out models/templates` 构建真实模板，后端通过 `TEMPLATE_DIR` 读取（docker-compose 已挂载 `models/templates`）。注意 Teeth3DS 是 CC BY-NC-ND 4.0，用它构建的模板不能直接商用；
- 底面：缺失牙场景（底面贴合牙龈）已实现；先把牙冠底边切成光滑的扇贝形再封底，真实口扫上导出的 STL 封闭且是流形；
- 预备体颈缘线：自动检测（几何方法，合成肩台预备体上平均误差约 0.13 mm）+ 编辑器人工编辑；上传时填写预备体牙号，或在编辑器里设置；
- 预备体牙冠：沿颈缘线精确切开预备体，组织面贴合（粘接间隙、颈缘处渐变为 0、去倒凹），外表面用对侧同名牙镜像，邻接距离按设计参数调整，保证最小冠厚度，颈缘处缝合成封闭 STL（`result/crown_<牙号>.stl`）。缺失牙同样做邻接调整。限制见 docs/algorithm-architecture.md 的 E2 一节。
