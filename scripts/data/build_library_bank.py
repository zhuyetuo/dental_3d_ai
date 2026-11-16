"""exocad 牙库 → 库牙表：每颗库牙（只要标准牙位 1–8.eoff，不要桥体、不要变体）的网格，和表面均匀采的 4000 个点（拟合用）。
输出 data/radial_lib/bank.npz：pts (N, 4000, 3)（牙库自己的坐标，每颗牙中心在原点）、verts/faces 拼在一起的网格（同一坐标，
第 i 颗是 verts[voff[i]:voff[i+1]]、faces[foff[i]:foff[i+1]]，下标从 0 起）、jaw (N,) 0=上颌 1=下颌、pos (N,) 1–8、name (N,)。
只读压缩包，不解压到磁盘。

    python scripts/data/build_library_bank.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
src = (HERE / "elosdent_eoff_index.py").read_text().split("\nlibs = defaultdict")[0]
ns = {"__name__": "lib"}
exec(compile(src.replace("/tmp/elos_peek.py", str(HERE / "elosdent_peek.py")).replace("scripts/data/elosdent_peek.py", str(HERE / "elosdent_peek.py")), "eoff_index", "exec"), ns)
parse, iterate = ns["parse_eoff"], ns["iterate"]

N_PTS = 4000
out = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "zyt/dental_3d_ai/data/radial_lib/bank.npz")
out.parent.mkdir(parents=True, exist_ok=True)
pts, jaw, pos, name, verts, faces = [], [], [], [], [], []
seen = set()
for bundle in ("bundle_02.7z", "bundle_03.zip", "bundle_04.rar", "bundle_05.rar", "bundle_08.rar"):
    k0 = len(pts)
    for n, buf in iterate(bundle):
        parts = n.split("/")
        if any("pontic" in p.lower() for p in parts):
            continue
        jaw_i = next((i for i, x in enumerate(parts) if x.lower() in ("upperjaw", "lowerjaw")), None)
        m = re.fullmatch(r"([1-8])\.eoff", parts[-1].lower())
        if jaw_i is None or jaw_i < 1 or not m or jaw_i != len(parts) - 2:
            continue
        lib = "/".join(parts[:jaw_i])
        key = (lib, parts[jaw_i].lower(), int(m.group(1)))
        if key in seen:  # 不同分包里的同一套牙库只要一份
            continue
        try:
            v, f = parse(buf)
        except Exception:  # noqa: BLE001
            continue
        mesh = trimesh.Trimesh(v, f, process=False)
        if mesh.area <= 0 or len(f) < 500:
            continue
        seen.add(key)
        p = np.asarray(trimesh.sample.sample_surface(mesh, N_PTS, seed=len(pts))[0], np.float32)
        c = p.mean(0)
        pts.append(p - c)
        verts.append((v - c).astype(np.float32))
        faces.append(f.astype(np.int32))
        jaw.append(0 if key[1] == "upperjaw" else 1)
        pos.append(key[2])
        name.append(f"{bundle}:{lib}")
    print(bundle, len(pts) - k0, flush=True)
voff = np.cumsum([0] + [len(v) for v in verts])
foff = np.cumsum([0] + [len(f) for f in faces])
np.savez_compressed(out, pts=np.stack(pts), verts=np.concatenate(verts), faces=np.concatenate(faces), voff=voff, foff=foff, jaw=np.array(jaw, np.int8), pos=np.array(pos, np.int8), name=np.array(name))
print("库牙", len(pts), "套", len(set(name)), "->", out)
for j in (0, 1):
    print("上颌" if j == 0 else "下颌", [int(((np.array(jaw) == j) & (np.array(pos) == q)).sum()) for q in range(1, 9)])
