# 分割评估与朝向归一化

## 1. 准备 Teeth3DS

1. **许可**：Teeth3DS 是 **CC BY-NC-ND 4.0**，只能用于非商业研究和教学。用于商业产品，包括训练或评估商业模型，需要向数据集作者单独申请商业授权（见 [3DTeethSeg22_challenge](https://github.com/abenhamadou/3DTeethSeg22_challenge) 的 “Dataset License and Commercial Use”）。
2. **下载**：数据在 OSF（https://osf.io/xctdy/），分 6 个部分，全部下载后合并成同一目录结构。
3. **划分文件**在挑战赛仓库的 `dataset/` 下：
   - `Teeth3DS_split/testing_lower.txt`、`testing_upper.txt`：官方划分，每行是 `病例_颌位`；
   - `3DTeethSeg_challenge_split/private-testing-set.txt`：挑战赛划分。
   TGNet 的权重用的是挑战赛的训练集，**评估 TGNet 要用挑战赛的测试集**，否则会有数据泄漏。

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

目标坐标系与 Teeth3DS / TGNet 一致：+Z 为解剖学上方（下颌牙冠朝 +Z，上颌牙冠朝 −Z），+Y 指向后方，+X 为患者左侧。

后端调用分割服务前默认会做朝向归一化（`SEGMENTER_ORIENT=1`），命令行用 `--no-orient` 关闭。标签是逐顶点的，所以不需要把结果变换回去，生成的牙也仍然在用户原来的坐标系里。

在合成牙弓上，任意随机旋转后，恢复误差 < 2°，单颌耗时约 1 秒。**在真实口扫上的效果还没有验证**：Teeth3DS 在 OSF 上，本环境访问不了。拿到数据后先跑 `--check-orientation`。

已知风险：
- 原始扫描如果被镜像过，左右会反；
- 带底座的封闭石膏模型，法向投票失效，会退化为“开放边界在牙龈侧”的判断；
- 大面积缺牙或严重不对称的牙列，对称面可能找偏。

`--check-orientation` 的 `flipped_rate`（误差 ≥ 90°，即方向整体反了）是最需要关注的数字。
