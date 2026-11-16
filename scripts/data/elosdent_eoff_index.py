"""exocad .eoff 牙库：解析网格（每个面 = uint16 顶点数 + int32 下标），统计所有分包里有多少套库、多少颗牙，每套的牙位和类型。
输出 ~/zyt/dental_data/.elosdent_inventory/eoff_index.json。只读压缩包，不解压到磁盘。"""
import json
import re
import struct
from collections import defaultdict
from pathlib import Path

import numpy as np

src = open("scripts/data/elosdent_peek.py").read().split("\nfor args in")[0]
ns = {}
exec(compile(src, "elos_peek", "exec"), ns)
c, INV, new, sf, sfm, openf, nxt, pname, psize, rd, free, fix = (ns[k] for k in
    ("c", "INV", "new", "sf", "sfm", "openf", "nxt", "pname", "psize", "rd", "free", "fix"))


def parse_eoff(buf):
    assert buf[:11] == b"OFF BINARY\n"
    nv, nf, _ = struct.unpack_from("<3i", buf, 11)
    v = np.frombuffer(buf, "<f4", nv * 3, 23).reshape(-1, 3)
    p = 23 + nv * 12
    faces = np.empty((nf, 3), np.int64)
    for i in range(nf):
        k = struct.unpack_from("<H", buf, p)[0]
        if k != 3:
            raise ValueError(f"面 {i} 有 {k} 个顶点")
        faces[i] = struct.unpack_from("<3i", buf, p + 2)
        p += 2 + 4 * k
    if faces.max() >= nv or faces.min() < 0:
        raise ValueError("下标越界")
    return v, faces


def iterate(bundle):
    """逐个读出包里的 .eoff（只读内存）。"""
    h = new(); sf(h); sfm(h)
    if openf(h, str(INV / bundle).encode(), 1 << 16) < 0:
        return
    e = c.c_void_p()
    try:
        while nxt(h, c.byref(e)) in (0, -20):
            raw = pname(e)
            if not raw:
                continue
            n = fix(raw.decode("utf-8", "replace")).replace("\\", "/")
            if not n.lower().endswith(".eoff"):
                continue
            size = psize(e)
            buf = c.create_string_buffer(size)
            got = 0
            while got < size:
                r = rd(e and h, c.byref(buf, got), size - got)
                if r <= 0:
                    break
                got += r
            yield n, buf.raw[:got]
    finally:
        free(h)


libs = defaultdict(lambda: {"teeth": {}, "pontics": {}, "bundle": None})
bad = []
for bundle in ("bundle_02.7z", "bundle_03.zip", "bundle_04.rar", "bundle_05.rar", "bundle_08.rar"):
    for n, buf in iterate(bundle):
        parts = n.split("/")
        jaw_i = next((i for i, x in enumerate(parts) if x.lower() in ("upperjaw", "lowerjaw")), None)
        if jaw_i is None or jaw_i < 1:
            continue
        lib = "/".join(parts[:jaw_i])           # 例：teeth/Shine 2、pontics/zirkonzahnSAO、zg3.0
        kind = "pontics" if "pontic" in lib.lower() else "teeth"
        name = re.sub(r"^(.*?/)?(teeth|pontics?)/", "", lib)
        m = re.match(r"(\d+)", parts[-1])
        pos = int(m.group(1)) if m else -1
        key = f"{bundle}:{name}"
        try:
            v, f = parse_eoff(buf)
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{n}: {exc}")
            continue
        ext = (v.max(0) - v.min(0)).round(1).tolist()
        libs[key]["bundle"] = bundle
        libs[key][kind][f"{parts[jaw_i][:5]}{pos}{'' if parts[-1][len(str(pos)):] in ('.eoff',) else '+' + parts[-1][len(str(pos)):-5]}"] = {
            "verts": len(v), "faces": len(f), "size_mm": ext}
    print(bundle, "done", flush=True)

summary = {k: {"bundle": d["bundle"], "teeth": len(d["teeth"]), "pontics": len(d["pontics"])} for k, d in libs.items()}
out = INV / "eoff_index.json"
out.write_text(json.dumps({"libraries": {k: dict(d) for k, d in libs.items()}, "bad": bad[:50]}, ensure_ascii=False))
full = [k for k, s in summary.items() if s["teeth"] >= 28]
print(f"\n牙库 {len(libs)} 套；完整（≥28 颗牙冠）的 {len(full)} 套；解析失败 {len(bad)} 个")
print("牙冠总数", sum(s["teeth"] for s in summary.values()), "桥体总数", sum(s["pontics"] for s in summary.values()))
for k in sorted(summary)[:60]:
    s = summary[k]
    print(f"  {k:60s} 牙冠 {s['teeth']:3d}  桥体 {s['pontics']:3d}")
print("失败样例", bad[:5])
