"""TeethSplineSeg（PTv3 + 牙弓 B 样条，我们用模拟缺牙数据微调过）分割服务，和 services/tgnet 同一个接口：

    POST /segment?jaw=lower   请求体：OBJ 文本（Teeth3DS 标准朝向）   返回：{"labels": [逐顶点 FDI]}
    GET  /health

在 TeethSplineSeg 自己的环境里跑（.venv-splineseg 或 docker/Dockerfile.splineseg），只用标准库起 HTTP。
第一版每个请求调一次作者的 prepare_dataset.py + inference.py（单个病例约半分钟，大半是加载模型和预处理）；同一时间只跑一个。

    SPLINESEG_DIR=third_party/TeethSplineSeg SPLINESEG_CKPT=<微调权重> python services/splineseg/server.py --port 8002
    python services/splineseg/server.py --test <obj> --jaw lower     # 不起服务，直接跑一个文件
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

SS = Path(os.environ.get("SPLINESEG_DIR", "/opt/TeethSplineSeg"))
CKPT = os.environ.get("SPLINESEG_CKPT", "/models/teethsplineseg/finetuned.ckpt")
DEVICE = os.environ.get("SPLINESEG_DEVICE", "0")  # cpu 也行（慢）
PY = os.environ.get("SPLINESEG_PY", sys.executable)
LOCK = threading.Lock()

# TeethSplineSeg 的类别号 1–16 → FDI（和 ml/dental_ai/external.py 的 SPLINESEG_UPPER / LOWER 一致，服务端不依赖 dental_ai）
SCHEME = json.loads(os.environ.get("SPLINESEG_SCHEME", "null") or "null")


def _scheme():
    if SCHEME:
        return SCHEME
    up = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
    return {"upper": up, "lower": [f + 20 for f in up]}


def count_vertices(obj: bytes) -> int:
    return sum(1 for line in obj.decode(errors="replace").splitlines() if line.startswith("v "))


def segment(obj: bytes, jaw: str) -> np.ndarray:
    """OBJ 文本 → 逐顶点 FDI（0 = 牙龈），顶点顺序和 OBJ 的 v 行一致。"""
    assert jaw in ("upper", "lower")
    with LOCK, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        case = "case"
        src = tmp / "src"
        (src / jaw / case).mkdir(parents=True)
        (src / ("upper" if jaw == "lower" else "lower")).mkdir()  # 作者的脚本要求两个颌的目录都在
        (src / jaw / case / f"{case}_{jaw}.obj").write_bytes(obj)
        n = count_vertices(obj)  # 预处理要有标注文件：全 0（我们的补丁让它在没有真值时也能跑）
        (src / jaw / case / f"{case}_{jaw}.json").write_text(json.dumps({"jaw": jaw, "labels": [0] * n, "instances": [0] * n}))
        for j in ("upper", "lower"):
            (src / f"testing_{j}.txt").write_text(f"{case}_{jaw}\n" if j == jaw else "")
            (src / f"training_{j}.txt").write_text("")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        t0 = time.time()
        subprocess.run([PY, "prepare_dataset.py", "--source", str(src), "--root", str(tmp / "root")], cwd=SS, env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        subprocess.run([PY, "inference.py", "--ckpt", CKPT, "--devices", DEVICE, "--root_folder", str(tmp / "root"), "--output_dir", str(tmp / "out"),
                        "--skip_eval", "--upsample_workers", "2", "--postprocess_workers", "1", "--num_workers", "0"],
                       cwd=SS, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        txt = tmp / "out" / "postprocessed" / f"{case}_{jaw}.txt"
        classes = np.loadtxt(txt, dtype=np.int64).reshape(-1)
        print(f"segment {jaw}: {len(classes)} 个顶点，{time.time() - t0:.1f} 秒", flush=True)
    if len(classes) != n:
        raise RuntimeError(f"预测 {len(classes)} 个顶点，OBJ 有 {n} 个")
    table = np.array([0] + _scheme()[jaw])
    valid = (classes >= 1) & (classes <= 16)
    return np.where(valid, table[np.clip(classes, 0, 16)], 0)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self._send(200, {"ok": True, "ckpt": CKPT, "device": DEVICE})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/segment":
            return self._send(404, {"error": "not found"})
        jaw = (parse_qs(u.query).get("jaw") or [""])[0]
        if jaw not in ("upper", "lower"):
            return self._send(400, {"error": "jaw 要是 upper 或 lower"})
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            labels = segment(body, jaw)
        except subprocess.CalledProcessError as exc:
            return self._send(500, {"error": f"TeethSplineSeg 失败：{exc.stderr.decode(errors='replace')[-800:]}"})
        except Exception as exc:  # noqa: BLE001
            return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
        self._send(200, {"labels": labels.tolist()})

    def log_message(self, fmt, *args):  # 只打印分割本身的耗时
        pass


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--test", type=Path, help="不起服务：直接分割这个 OBJ")
    p.add_argument("--jaw", default="lower")
    args = p.parse_args()
    if args.test:
        try:
            labels = segment(args.test.read_bytes(), args.jaw)
        except subprocess.CalledProcessError as exc:
            print(exc.stderr.decode(errors="replace")[-3000:])
            raise
        print("牙号：", sorted(int(v) for v in np.unique(labels) if v))
        return
    print(f"TeethSplineSeg 服务：{args.host}:{args.port}，权重 {CKPT}，设备 {DEVICE}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
