import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
import trimesh

from dental_ai.synthetic import synthetic_arch


@pytest.fixture(scope="session")
def fake_segmenter_url():
    """模拟分割服务：按顶点顺序返回合成牙弓的真值标签（与真实服务的契约一致：标签与收到的顶点一一对应）。

    收到的网格可能已经做过朝向归一化，坐标会变，但顶点顺序不变。"""
    mesh, labels = synthetic_arch("lower", missing=(36,))

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            jaw = parse_qs(urlparse(self.path).query).get("jaw", [""])[0]
            body = self.rfile.read(int(self.headers["Content-Length"]))
            received = trimesh.load(trimesh.util.wrap_as_stream(body), file_type="obj", process=False, force="mesh")
            if jaw != "lower":
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"only lower")
                return
            if len(received.vertices) != len(labels):
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"unexpected vertex count")
                return
            out = labels.tolist()
            data = json.dumps({"labels": out}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
