# 分割评估与朝向归一化

## 1. 准备 Teeth3DS

1. **许可**：Teeth3DS 是 **CC BY-NC-ND 4.0**，只能用于非商业研究和教学。用于商业产品，包括训练或评估商业模型，需要向数据集作者单独申请商业授权（见 [3DTeethSeg22_challenge](https://github.com/abenhamadou/3DTeethSeg22_challenge) 的 “Dataset License and Commercial Use”）。
2. **下载**：数据在 OSF（https://osf.io/xctdy/），分 6 个部分，全部下载后合并成同一目录结构。`python scripts/download_datasets.py teeth3ds` 会通过 OSF API 下载并解压（未实测，开发环境访问不了 OSF）；`teeth3ds-sample` 只下载官方仓库里的 1 个示例病例，`teeth3ds-splits` 下载划分文件。
   - 注意：真实 OBJ 里约 7% 的顶点没有被三角形引用，标签按全部 `v` 行编号。用 `dental_ai.io.load_mesh` 读取（trimesh 会丢掉这些顶点）。
3. **划分文件**在挑战赛仓库的 `dataset/` 下：
   - `Teeth3DS_split/testing_lower.txt`、`testing_upper.txt`：官方划分，每行是 `病例_颌位`；
   - `3DTeethSeg_challenge_split/private-testing-set.txt`：挑战赛划分。
   TGNet 的权重用的是挑战赛的训练集，**评估 TGNet 要用挑战赛的测试集**，否则会有数据泄漏。
   **两套划分大量交叉**（2026-11-01 用本地划分文件统计）：各 1200 训练 / 600 测试，Teeth3DS 测试集里 426 个扫描在挑战赛训练集里，
   挑战赛测试集里 426 个在 Teeth3DS 训练集里。所以用 Teeth3DS 划分训练的模型（我们的 CPU 基线、深度模型、TeethSplineSeg）不能在挑战赛测试集上评估，反过来也一样。
   **不同来源的模型放在一起比较，用两个测试集的交集**（174 个扫描）：`python -m dental_ai.external fair-split`，GPU 脚本导出到 `data/fair_test`。

目录结构：

```
<data-dir>/<病例>/<病例>_lower.obj
<data-dir>/<病例>/<病例>_upper.obj
<gt-dir>/<病例>/<病例>_lower.json
<gt-dir>/<病例>/<病例>_upper.json
```

## 2. 运行评估

先装好算法包：

```bash
pip install -e ml
```

**评估已有的预测结果**（比如 TGNet 自带 `start_inference.py` 的输出，平铺的 `<病例>_<颌位>.json`）：

```bash
python -m dental_ai.evaluation --data-dir teeth3ds/obj --gt-dir teeth3ds/gt --split private-testing-set.txt --pred-dir tgnet_out --out results/tgnet
```

**直接调用分割服务**（先 `docker compose --profile gpu up`）。Teeth3DS 本身已经是标准朝向，加 `--orient` 可以同时检验“朝向归一化 + 分割”整条链路：

```bash
python -m dental_ai.evaluation --data-dir teeth3ds/obj --gt-dir teeth3ds/gt --split private-testing-set.txt --segmenter-url http://localhost:8001 --orient --out results/tgnet_oriented
```

**只验证朝向归一化**（不需要 GPU）：

```bash
python -m dental_ai.evaluation --data-dir teeth3ds/obj --gt-dir teeth3ds/gt --limit 100 --check-orientation --out results/orientation
```

`--limit N` 只评估前 N 个扫描，适合快速检查。

**衡量编号纠错的效果**：同一批预测分别加、不加 `--fix-numbering` 各跑一次，对比 `tir` 和 `all_correct`：

```bash
python -m dental_ai.evaluation --data-dir teeth3ds/obj --gt-dir teeth3ds/gt --split private-testing-set.txt --pred-dir tgnet_out --fix-numbering --out results/tgnet_fixed
```

## 3. 输出

| 文件 | 内容 |
|---|---|
| `summary.json` | 全部 / 上颌 / 下颌的平均指标；每个牙号的平均 IoU；失败的扫描；按官方规则把失败记 0 后的指标 |
| `per_scan.csv` | 每个扫描的全部指标和推理耗时 |
| `per_tooth_iou.csv` | 每个扫描每颗牙的 IoU，可以用来找哪类牙最差 |
| `orientation.csv`、`orientation_summary.json` | 朝向检查：数据是否符合坐标约定、随机旋转后的恢复误差 |

## 4. 指标

| 指标 | 口径 | 说明 |
|---|---|---|
| `tla` | 挑战赛官方 | exp(-TLA)，牙齿定位 |
| `tsa` | 挑战赛官方 | 牙齿 / 牙龈二分类的 micro-F1（等于顶点二分类准确率） |
| `tir` | 挑战赛官方 | 匹配上的牙中，位置足够近且编号正确的比例 |
| `score` | 挑战赛官方 | 三项平均；排行榜第一（CGIP）为 0.9539 |
| `inst_iou`、`inst_f1`、`inst_tir` | TGNet | 和 TGNet README 的 IoU / TSA / CLS 对比 |
| `t_miou` | 论文 | 每颗真值牙的 IoU 平均（3DTeethSAM 报告 91.90%） |
| `all_correct` | 论文 | 整副牙列全对率（TSegAgent 在私有集上为 51.96%） |

挑战赛指标严格复刻了官方 `evaluation/evaluation.py`（commit 6220bfa）的实现，包括它的几个特殊之处：
- 真值牙和预测牙按质心用匈牙利算法一一匹配；
- “牙齿尺寸”是每个坐标轴上 sqrt(Σ(顶点 − 质心)²)；
- 没有匹配上的真值牙记 5 倍惩罚；
- 编号不唯一的预测实例当作牙龈。

**还没做的验证**：我们的实现还没有和官方代码在同一批数据上直接比对，原因是本环境不允许执行下载来的外部代码。拿到数据后，建议先用官方的 `evaluation.py` 和本模块跑同一批预测，确认数字一致。

**看哪个指标**：选模型、决定上线时，以**自有跨来源测试集**上的 `all_correct` 和 `t_miou` 为主，`score` 用来和论文、排行榜对比。

## 5. 朝向归一化

`ml/dental_ai/orientation.py`，不依赖分割结果：

1. **咬合方向**：顶点分布的最小主成分。单颌口扫是一块马蹄形的“薄片”；方向符号由面法向按面积加权投票决定；
2. **左右方向**：在咬合平面内搜索左右镜像最对称的方向（用 3D 点，牙齿的高度起伏是主要信息）；
3. **前后方向**：最前端的点（切牙）靠近中线，最后端的点（磨牙）分布在两侧。

目标坐标系与 Teeth3DS 原始数据（TGNet 的训练坐标系）一致：**上下颌都是牙冠朝 +Z**，+Y 指向后方，X = Y × Z（下颌是患者左侧，上颌是患者右侧——上颌模型是“翻过来、从咬合面看”的）。

以前写的是“上颌牙冠朝 −Z、+X 都是患者左侧”，在 Teeth3DS 的上颌扫描上核对是错的（相差绕前后轴 180°），按旧约定转过去再交给 TGNet，上颌分割会出错，已改正。

后端调用分割服务前默认会做朝向归一化（`SEGMENTER_ORIENT=1`），命令行用 `--no-orient` 关闭。标签是逐顶点的，所以不需要把结果变换回去，生成的牙也仍然在用户原来的坐标系里。

`--check-orientation` 把归一化结果与“用真值标签算出的标准坐标轴”（`evaluation.label_frame`：牙冠方向、左右同名牙对、X = Y × Z）比较，不依赖数据原来的朝向，所以原始坐标系不同的数据集也能用。

真实数据上（2026-10-31，每个扫描随机旋转 3 次取最大误差）：

| 数据 | 扫描数 | 误差中位数 | 10° 以内 | 整体反向 |
|---|---|---|---|---|
| Teeth3DS 官方测试集（随机 100 个） | 100 | 3.8° | 97% | 0 |
| Poseidon3D test-simple（带底座的封闭模型） | 20 | 4.4° | 90% | 1 |
| Poseidon3D test-complex（拥挤、缺牙） | 19 | 9.6° | 53% | 1 |

合成牙弓上误差 < 2°，单颌耗时约 1 秒。

已知风险：
- 原始扫描如果被镜像过，左右会反；
- 带底座的封闭模型（Poseidon3D）：法向投票为 0，改用“底座底面是大片朝外的平面”判断牙冠方向（实测底座一侧平面占表面积 16–24%，牙冠一侧 2–4%），前后方向只用牙冠一侧的一半高度判断（底座轮廓会让判断反过来）。改之前 Poseidon3D 全部反向；
- 大面积缺牙或严重不对称的牙列，对称面可能找偏。

`--check-orientation` 的 `flipped_rate`（误差 ≥ 90°，即方向整体反了）是最需要关注的数字。
