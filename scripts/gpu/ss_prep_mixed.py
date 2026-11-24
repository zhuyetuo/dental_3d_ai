"""把剩下没预处理的样本分给 GPU 进程和 CPU 进程（按速度加权），写出每个进程的列表目录 /tmp/ssaug2/src<i>。
GPU 进程排在前面（0..G-1），CPU 进程在后面。打印 G 和总数。"""
import os
import sys
from pathlib import Path

SRC = Path(sys.argv[1]); OLD = Path(sys.argv[2]); G, C = int(sys.argv[3]), int(sys.argv[4])
W = Path("/tmp/ssaug2")
todo = []
for jaw in ("lower", "upper"):
    for n in (SRC / f"training2_{jaw}.txt").read_text().split():
        if not (OLD / f"data_{n}.pt").exists():
            todo.append((jaw, n))
# GPU 进程每个约 50 秒/样本（12 个挤一张卡），CPU 进程约 150 秒/样本 → 权重 3 : 1
weights = [3.0] * G + [1.0] * C
tot = sum(weights)
lists = [[] for _ in weights]
acc = [0.0] * len(weights)
for item in todo:  # 每次给“相对负担”最小的进程
    k = min(range(len(weights)), key=lambda i: acc[i] / weights[i])
    lists[k].append(item); acc[k] += 1
for i, l in enumerate(lists):
    d = W / f"src{i}"; d.mkdir(parents=True, exist_ok=True)
    for jaw in ("lower", "upper"):
        link = d / jaw
        if not link.exists():
            os.symlink(SRC / jaw, link)
        (d / f"testing_{jaw}.txt").write_text("".join(n + "\n" for j, n in l if j == jaw))
        (d / f"training_{jaw}.txt").write_text("")
print(len(todo), [len(l) for l in lists[:2]], [len(l) for l in lists[-2:]])
