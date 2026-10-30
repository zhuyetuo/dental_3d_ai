"""通过 HTTP 调用独立部署的分割服务（如 services/tgnet）。

分割模型各自依赖不同版本的 PyTorch 和 CUDA 算子，放在单独的容器里，主程序只依赖这个 HTTP 接口：

    POST {url}/segment?jaw=<upper|lower>   请求体：OBJ 文本   返回：{"labels": [...]}
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import numpy as np
import trimesh

from ..fdi import Jaw
from .base import Segmenter


class RemoteSegmenter(Segmenter):
    def __init__(self, url: str, name: str = "remote", timeout: float = 300.0):
        self.url = url.rstrip("/")
        self.name = name
        self.timeout = timeout

    def predict(self, mesh: trimesh.Trimesh, jaw: Jaw) -> np.ndarray:
        # OBJ 按原顺序写出顶点，标签与 mesh.vertices 一一对应
        body = mesh.export(file_type="obj").encode()
        req = urllib.request.Request(
            f"{self.url}/segment?jaw={jaw}",
            data=body,
            headers={"Content-Type": "text/plain"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                payload = json.loads(res.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"分割服务返回错误 {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"无法连接分割服务 {self.url}: {exc.reason}") from None

        labels = np.asarray(payload["labels"], dtype=np.int64)
        if len(labels) != len(mesh.vertices):
            raise RuntimeError(f"分割服务返回 {len(labels)} 个标签，网格有 {len(mesh.vertices)} 个顶点")
        return labels
