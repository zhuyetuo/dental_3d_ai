"""TGNet 分割服务。

在 TGNet 自己的环境（旧版 PyTorch + 自定义 CUDA 算子 pointops）里运行，对外只暴露 HTTP：

    POST /segment?jaw=lower   请求体：OBJ 文本    返回：{"labels": [逐顶点 FDI], "instances": [...]}
    GET  /health

和 TGNet 自带的 start_inference.py 相比：
- 模型只在启动时加载一次，不用每个病例重新加载；
- 颌位从参数传入，不依赖 “<病例>_<颌>.obj” 这种文件命名；
- TGNet 内部会合并重复顶点，返回的标签数可能少于输入顶点数，这里映射回输入的顶点顺序。

输入要求（TGNet 训练数据的坐标系）：Y 轴指向后方，上下颌的 Z 轴方向相同。
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from typing import Callable

import numpy as np
import trimesh
from fastapi import FastAPI, HTTPException, Query, Request

# predict(obj 路径, 颌位) -> 去重后顶点的 FDI 标签
Predictor = Callable[[str, str], np.ndarray]


def load_tgnet_predictor() -> Predictor:
    repo = os.environ.get("TGNET_DIR", "/opt/ToothGroupNetwork")
    ckpt_fps = os.environ.get("TGNET_CKPT_FPS", "/models/tgnet_fps.h5")
    ckpt_bdl = os.environ.get("TGNET_CKPT_BDL", "/models/tgnet_bdl.h5")
    for path in (ckpt_fps, ckpt_bdl):
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到权重文件 {path}，请把 TGNet 的 ckpts 放到挂载的 models 目录")
    sys.path.insert(0, repo)
    from inference_pipelines.inference_pipeline_maker import make_inference_pipeline

    # 注意：start_inference.py 会自动给路径加 .h5，这里直接传完整路径
    pipeline = make_inference_pipeline("tgnet", [ckpt_fps, ckpt_bdl])

    def predict(obj_path: str, jaw: str) -> np.ndarray:
        sem = np.asarray(pipeline(obj_path)["sem"]).astype(np.int64)
        # 模型输出上颌编号 11–28；下颌整体加 20 得到 31–48（与 TGNet 的 ScanSegmentation 一致）
        if jaw == "lower":
            sem[sem > 0] += 20
        return sem

    return predict


def expand_to_input_vertices(vertices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """TGNet 用 open3d 的 remove_duplicated_vertices 去重（保留首次出现的顶点、保持原顺序）。
    把去重后的标签映射回原始顶点。"""
    if len(labels) == len(vertices):
        return labels
    _, first, inverse = np.unique(vertices, axis=0, return_index=True, return_inverse=True)
    if len(first) != len(labels):
        raise ValueError(f"标签数 {len(labels)} 与去重后顶点数 {len(first)} 不一致")
    rank = np.argsort(np.argsort(first))  # 每个唯一顶点在去重序列中的位置
    return labels[rank[inverse.reshape(-1)]]


def create_app(load_predictor: Callable[[], Predictor] = load_tgnet_predictor) -> FastAPI:
    app = FastAPI(title="TGNet segmentation service")
    state: dict = {}
    lock = threading.Lock()  # 单 GPU，串行推理

    def predictor() -> Predictor:
        with lock:
            if "predict" not in state:
                state["predict"] = load_predictor()
        return state["predict"]

    @app.get("/health")
    def health() -> dict:
        try:
            predictor()
        except Exception as exc:  # 权重缺失等，返回原因而不是 500
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "model": "tgnet"}

    @app.post("/segment")
    async def segment(request: Request, jaw: str = Query(..., pattern="^(upper|lower)$")) -> dict:
        body = await request.body()
        if not body:
            raise HTTPException(400, "请求体为空，需要 OBJ 文本")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, f"case_{jaw}.obj")
            with open(path, "wb") as f:
                f.write(body)
            try:
                mesh = trimesh.load(path, file_type="obj", process=False, force="mesh")
            except Exception as exc:
                raise HTTPException(400, f"无法解析 OBJ: {exc}") from None
            predict = predictor()
            with lock:
                labels = predict(path, jaw)
        try:
            labels = expand_to_input_vertices(np.asarray(mesh.vertices), np.asarray(labels))
        except ValueError as exc:
            raise HTTPException(500, str(exc)) from None
        return {"labels": labels.tolist(), "model": "tgnet"}

    return app


app = create_app()
