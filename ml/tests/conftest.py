import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
import trimesh
from scipy.spatial import cKDTree

from dental_ai.synthetic import synthetic_arch


@pytest.fixture(scope="session")
def fake_segmenter_url():
    """模拟分割服务：按顶点坐标查合成牙弓的真值标签，用来测试 HTTP 调用链路。"""
    mesh, labels = synthetic_arch("lower", missing=(36,))
    tree = cKDTree(mesh.vertices)

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
            out = labels[tree.query(received.vertices)[1]].tolist()
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
