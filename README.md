# dental_3d_ai
**口腔 3D 扫描模型（上下颌 STL）输入 → 牙齿实例分割 → 识别缺牙位置 → 自动生成缺失牙齿 3D 模型**

整体方案、技术选型和参考论文见 [docs/architecture.md](docs/architecture.md)；竞品解析（dentalai.ai / 3Shape / 先临三维）和功能规划见 [docs/product-analysis.md](docs/product-analysis.md)；产业界、学术界格局和 AI 提效数据见 [docs/industry-landscape.md](docs/industry-landscape.md)。

## 目录

```
ml/        核心算法包 dental_ai（与 Web 无关，可单独使用）
  dental_ai/fdi.py              FDI 牙位编号工具
  dental_ai/segmentation/       分割接口、TGNet 适配器、后处理
  dental_ai/detection/          缺牙检测
  dental_ai/generation/         缺失牙生成（V1：对侧镜像 + 缺隙拟合）
  dental_ai/margin/             牙颈线提取、贴合牙龈的底面生成
  dental_ai/preferences.py      设计参数（邻接距离、咬合距离、粘接间隙等）
  dental_ai/pipeline.py         端到端流程 / 命令行入口
  dental_ai/synthetic.py        合成牙弓，用于测试和联调
backend/   FastAPI 服务
frontend/  React + Three.js 查看器
scripts/   工具脚本
docs/      设计文档
```

## 快速开始

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

运行测试：

```bash
cd ml && pytest -q
```

```bash
cd backend && pytest -q
```

## 当前状态

- 分割：目前读取 Teeth3DS 格式的标签文件；TGNet 推理适配器已写好（`segmentation/tgnet.py`），需要在它自己的 Docker 环境中运行；
- 带标签的数据请用 OBJ/PLY。STL 没有共享顶点，读入时会合并顶点，顶点顺序与标签文件对不上；
- 生成：V1 对侧镜像，只处理两侧都有邻牙的缺隙；
- 底面：缺失牙场景（底面贴合牙龈）已实现；预备体颈缘线检测待做。
