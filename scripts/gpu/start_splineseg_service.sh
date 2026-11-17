#!/bin/bash
# 起 TeethSplineSeg 分割服务（宿主机上，TeethSplineSeg 自己的环境），只监听 docker 网络的网关地址，worker 容器能访问、局域网访问不到
cd ~/zyt/dental_3d_ai
pkill -f '[s]ervices/splineseg/server.py' && sleep 2
export SPLINESEG_DIR=$PWD/third_party/TeethSplineSeg SPLINESEG_CKPT=$PWD/models/teethsplineseg/finetuned_aug.ckpt SPLINESEG_DEVICE=0 SPLINESEG_PY=$PWD/.venv-splineseg/bin/python
GW=$(docker network inspect dental_3d_ai_default --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}')
nohup setsid .venv-splineseg/bin/python services/splineseg/server.py --host $GW --port 8002 > runs/splineseg_server.log 2>&1 < /dev/null &
sleep 3
curl -s http://$GW:8002/health; echo
# 端到端：一个 Bite2Text 病例，两个模型各分一遍，看选了哪个
.venv/bin/python - "$GW" <<'EOF'
import sys, io, zipfile
import numpy as np, trimesh
sys.path.insert(0, "ml")
from dental_ai.segmentation.deep import DeepSegmenter
from dental_ai.segmentation.remote import RemoteSegmenter
from dental_ai.segmentation.fusion import FusionSegmenter
from dental_ai.orientation import OrientedSegmenter
z = zipfile.ZipFile("/home/zyt/zyt/dental_data/Bite2Text.zip")
mesh = trimesh.load(io.BytesIO(z.read("F1991/ios/ios_lower.stl")), file_type="stl")
f = FusionSegmenter(DeepSegmenter("models/segmentation/deep.pt"), OrientedSegmenter(RemoteSegmenter(f"http://{sys.argv[1]}:8002", name="teethsplineseg")), name="fusion")
lab = f.predict(mesh, "lower")
print("选择：", f.last, "牙号：", sorted(int(v) for v in np.unique(lab) if v))
EOF
tail -2 runs/splineseg_server.log
