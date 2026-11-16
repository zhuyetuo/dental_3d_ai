"""elosdent 数据的 10 个分包：文件名还原成中文、按目录分组统计类型、找出完整病例的候选。只读清单（Codex 生成的 bundle_*.json），不解压。"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

INV = Path.home() / "zyt/dental_data/.elosdent_inventory"


def fix(name: str) -> str:
    """zip 里的 GBK 文件名被当成 cp437 读出来了：还原。"""
    try:
        return name.encode("cp437").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


KEY = re.compile(r"upper|lower|jaw|prep|crown|cad|scan|antag|opp|margin|abut|implant|pontic|bridge|die|上颌|下颌|牙冠|基台|预备|桥|种植|代型", re.I)
for j in sorted(INV.glob("bundle_*.json")):
    entries = json.loads(j.read_text())
    arch = next((p.name for p in INV.glob(j.stem + ".*") if p.suffix != ".json"), "?")
    paths = [(PurePosixPath(fix(n).replace("\\", "/")), s) for n, s in entries]
    total = sum(s for _, s in paths)
    print(f"\n######## {arch}：{len(paths)} 个文件，{total / 2**20:.0f} MB")
    print("  类型：", Counter(p.suffix.lower() or "(无)" for p, _ in paths).most_common(12))
    # 按前两级目录分组
    groups = defaultdict(list)
    for p, s in paths:
        key = "/".join(p.parts[:2]) if len(p.parts) > 2 else (p.parts[0] if len(p.parts) > 1 else "(根)")
        groups[key].append((p, s))
    print(f"  前两级目录 {len(groups)} 个，例：")
    for k in list(groups)[:6]:
        g = groups[k]
        print(f"    {k}  [{len(g)} 个，{Counter(p.suffix.lower() for p, _ in g).most_common(5)}]")
    # 文件名里的关键词
    words = Counter(w.lower() for p, _ in paths for w in set(KEY.findall(str(p))))
    print("  关键词：", words.most_common(15))
    # 每个目录：有没有口扫 + 牙冠的组合
    dirs = defaultdict(set)
    for p, _ in paths:
        n = p.name.lower()
        d = str(p.parent)
        if n.endswith((".stl", ".ply", ".obj", ".dcm", ".eoff", ".dme")):
            if re.search(r"upper|lower|jaw|上颌|下颌", n):
                dirs[d].add("jaw")
            if re.search(r"crown|cad|牙冠|anatomic", n):
                dirs[d].add("crown")
            if re.search(r"prep|预备|die|代型", n):
                dirs[d].add("prep")
            if re.search(r"antag|opp|对颌", n):
                dirs[d].add("antag")
    full = [d for d, k in dirs.items() if {"jaw", "crown"} <= k]
    print(f"  同一目录里有口扫 + 牙冠的：{len(full)} 个", full[:5])
    print("  样例路径：")
    for p, s in paths[:: max(1, len(paths) // 8)][:8]:
        print(f"    {s / 1e6:7.2f} MB  {p}")
