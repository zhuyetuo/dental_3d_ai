# 网页演示用的示例病例

`scans/` 里是 3 个**未经修改**的 Teeth3DS 口扫（OBJ 与逐顶点标注 JSON，只做了 gzip 压缩），用于网页上的“示例病例”。

## 数据来源和许可

- 数据集：**Teeth3DS**（3DTeethSeg'22 挑战赛数据集），Ben-Hamadou A. 等，*Teeth3DS: a benchmark for teeth segmentation and labeling from intra-oral 3D scans*，arXiv:2210.06094；
  数据主页 https://osf.io/xctdy/ ，挑战赛仓库 https://github.com/abenhamadou/3DTeethSeg22_challenge
- 许可：**CC BY-NC-ND 4.0**（全文见 `LICENSE-Teeth3DS.txt`，https://creativecommons.org/licenses/by-nc-nd/4.0/ ）。
  只能用于非商业目的；用于商业产品（包括训练或评估商业模型）需向数据集作者申请商业授权（contact@udini.ai）。
- 扫描：`01KTSH6A_lower`、`016N696F_upper`、`RXKYWHG6_lower`，均来自 Teeth3DS 官方测试集划分，内容未改动。

## 为什么不放修改过的扫描

ND（禁止演绎）不允许分发修改后的版本。“去掉几颗牙”“磨成预备体”这些演示场景只在 `demos.json` 里写配置，
由后端在本地加载示例时现场生成（`backend/app/demos.py`，用 `dental_ai.synthetic.simulate_missing / simulate_preparation`），
不会提交到仓库。

重新生成本目录：`python scripts/make_demos.py --data-dir <导出的 Teeth3DS 扫描目录>`。
