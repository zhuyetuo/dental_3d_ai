# 公开数据集清单与获取方法

本开发环境（云端会话）的网络策略访问不了 osf.io、zenodo.org、huggingface.co、figshare.com，能访问 GitHub 和 Google Cloud Storage。
所以做法是：**在你自己的电脑上手动下载 → 上传到你自己的 GCS 私有桶 → 这里用只读服务账号取下来**。

所有数据都放在 `data/` 下（已在 .gitignore 中），**不进 git，不分发**。

## 1. 清单

| 数据集 | 内容 | 格式 | 许可 | 获取 | 在本项目里的用途 | 状态 |
|---|---|---|---|---|---|---|
| **Teeth3DS / 3DTeethSeg'22** | 下载到的 7 个分卷共 **1900 个口扫（上下颌各 950）**，逐顶点 FDI 标注；扫描仪 Primescan、Trios3、iTero | 每个扫描一个 OBJ + 一个 JSON（`labels`、`instances`，逐顶点），在 zip 里同一个目录 `<lower|upper>/<病例>/`；约 7% 的顶点没有被三角形引用（见 [real-data-test.md](real-data-test.md)）；坐标系：上下颌都是牙冠朝 +Z、磨牙在 +Y，上颌是“翻过来”的（见 evaluation.md 第 5 节） | **CC BY-NC-ND 4.0**，仅限非商业；训练或评估商业模型、用于商业产品需向 contact@udini.ai 申请商业授权 | [OSF xctdy](https://osf.io/xctdy/)；划分文件在 [官方仓库](https://github.com/abenhamadou/3DTeethSeg22_challenge) `dataset/` | 分割评估（官方指标）、朝向归一化验证、编号纠错统计、缺牙生成评估、模板库构建 | **已下载**（用户的 Google Drive，2026-10-31，8.9 GB zip，解压约 31 GB）；`Teeth3DS+/license.txt` 确认是 CC BY-NC-ND 4.0 |
| **Teeth3DS+ / 3DTeethLand'24** | 在 Teeth3DS 上增加牙齿关键点（牙尖、近远中点等），训练 240 个、测试约 100 个扫描 | 每个扫描一个 `<病例>_<颌位>__kpt.json` | CC BY-NC-ND 4.0（同一个 license.txt） | 见 [arXiv 2210.06094](https://arxiv.org/abs/2210.06094)、[arXiv 2512.08323](https://arxiv.org/html/2512.08323v1) 和挑战赛页面 | 关键点（C1）、牙齿局部坐标系、咬合面 | **已下载**（同上，关键点文件还没解析） |
| **Poseidon3D** | 200 个正畸病例（每个一个颌），包含牙列拥挤、缺牙等困难情况（Kubík & Španěl，Bioengineering 2024）；划分：test-simple 20、test-complex 20、validation 10、train 150 | 官方没有说明，从文件推出来的：`<病例>_MODEL_mandible|maxilla.stl`（**带底座的封闭模型**，不是开放的口扫曲面）+ `<病例>_REGIONS_FACES_*.pkl`（逐三角形区域号）+ `<病例>_MAPPING_*.pkl`（牙位 → 区域号）；位置 → FDI 的对应用 138 个自带 FDI 标注的病例核对过，全部一致。**左右手性与 Teeth3DS 相反**，导出时镜像。pkl 用禁止加载类的 Unpickler 读取 | 待核对：仓库没有 LICENSE 文件，README 写“intended for research purposes”；以 Zenodo 页面上的许可为准 | [Zenodo 15608906](https://zenodo.org/records/15608906)；代码仓库 [tiborkubik/lmvseg-poseidon3d](https://github.com/tiborkubik/lmvseg-poseidon3d)（commit `7cb29c2`，只有 README 和示意图） | 困难病例的分割评估（拥挤、缺牙）、编号纠错、缺牙检测 | **已下载**（用户的 Google Drive，1.0 GB zip）；许可仍待核对（压缩包和 metadata 里都没有） |
| 先临三维牙齿分割数据集 | IEEE DataPort，DOI 10.21227/txce-st96 | 待核对 | 待核对 | [IEEE DataPort](https://ieee-dataport.org/documents/shining3d-tooth-segmentation-dataset)（需要 IEEE 账号） | 跨扫描仪（先临）评估 | 未下载 |
| ODD | ToothCraft 训练用的完整牙弓数据 | 待核对 | 待核对 | 见 [ToothCraft 仓库](https://github.com/ikarus1211/VISAPP_ToothCraft) | 缺牙生成 | 未下载 |

“待核对”的项需要你下载时在下载页面上看一下（尤其是许可），告诉我后我更新这张表。

### 导出成统一格式

数据放在 `data/drive/raw/`（`download_datasets.py fetch --src gdrive://...`）。全部解压要 40 GB，按需导出成 Teeth3DS 目录结构（OBJ + 逐顶点 JSON），之后评估、模板库、流程、网页都能直接用：

```bash
cd ml && python -m dental_ai.datasets teeth3ds --zips ../data/drive/raw/Teeth3DS --split <划分文件> --out ../data/teeth3ds_test --limit 100
```

```bash
cd ml && python -m dental_ai.datasets poseidon3d --zip ../data/drive/raw/Poseidon3D/poseidon3d.zip --split test-complex --out ../data/poseidon3d/test-complex
```

## 2. 手动下载

### Teeth3DS（优先）
1. 打开 https://osf.io/xctdy/ ，下载全部 6 个部分（zip），**不用解压**；
2. 划分文件不用下载，这里能直接从 GitHub 取（`download_datasets.py teeth3ds-splits`）。

### Poseidon3D
1. 打开 https://zenodo.org/records/15608906 ，下载全部文件；
2. 记下页面上的许可（License）和文件格式说明。

## 3A. 放在 Google Drive（更简单）

云端环境访问不了 `drive.google.com`、`drive.usercontent.google.com`，但 Drive API（`www.googleapis.com`）可以访问。做法：

1. Drive 文件夹（含子文件夹）共享设置为“**知道链接的任何人** → 查看者”；
2. 在 Google Cloud 控制台：选一个项目 → “API 和服务” → 启用 **Google Drive API** → “凭据” → 创建 **API 密钥**；
   建议在密钥的“API 限制”里只勾选 Google Drive API；
3. 把 API 密钥放到云端环境的环境变量 `GOOGLE_API_KEY`（环境设置里加，**不要贴到聊天里**），新开会话；
4. 我运行（文件夹 ID 是链接里 `folders/` 后面那一段）：

```bash
python scripts/download_datasets.py fetch --name drive_datasets --src gdrive://<文件夹 ID>
```

会递归下载整个文件夹到 `data/drive_datasets/raw/`（保持子文件夹结构），zip 自动解压。下载日志不打印密钥。

另一种办法：在环境的网络设置里放行 `drive.google.com` 和 `drive.usercontent.google.com`，再用 gdown 下载（不需要 API 密钥）。

## 3. 上传到 Google Cloud Storage

建议的目录结构（文件保持下载时的样子，zip 不用解压）：

```
gs://<你的桶>/datasets/teeth3ds/      6 个 zip
gs://<你的桶>/datasets/poseidon3d/    Zenodo 上下载的全部文件
```

在你电脑上（已安装 Google Cloud SDK 并登录）：

```bash
gcloud storage buckets create gs://<你的桶> --location=asia-east1 --uniform-bucket-level-access --public-access-prevention
```

```bash
gcloud storage cp ~/Downloads/teeth3ds/*.zip gs://<你的桶>/datasets/teeth3ds/
```

桶保持**私有**（上面的 `--public-access-prevention`）：这些数据集都只允许非商业研究使用，有的禁止二次分发。

## 4. 给云端会话只读权限

创建一个只能读这个桶的服务账号，生成密钥：

```bash
gcloud iam service-accounts create dental-data-reader --display-name="dental datasets read-only"
```

```bash
gcloud storage buckets add-iam-policy-binding gs://<你的桶> --member="serviceAccount:dental-data-reader@<你的项目ID>.iam.gserviceaccount.com" --role="roles/storage.objectViewer"
```

```bash
gcloud iam service-accounts keys create dental-data-reader.json --iam-account=dental-data-reader@<你的项目ID>.iam.gserviceaccount.com
```

然后把 `dental-data-reader.json` 的**全部内容**放到云端环境的环境变量 `GCS_SERVICE_ACCOUNT_JSON` 里：会话标题栏的云端环境菜单 → Edit → 环境变量（或 Network secrets）。
**不要把密钥贴到聊天里。** 新开的会话才会读到这个变量。用完可以在 GCP 控制台删除这个密钥。

网络：`storage.googleapis.com`、`oauth2.googleapis.com` 在本环境已实测可访问，不需要额外放行。

## 5. 在云端取数据

告诉我桶的路径，我会运行：

```bash
python scripts/download_datasets.py fetch --name teeth3ds --src gs://<你的桶>/datasets/teeth3ds/
```

它会先用 `GCS_SERVICE_ACCOUNT_JSON` 登录 gsutil，把文件拷到 `data/teeth3ds/raw/`，zip 解压到 `data/teeth3ds/extracted/`。
之后评估、模板库等命令的 `--data-dir` 和 `--gt-dir` 都可以直接给 `data/teeth3ds`：OBJ 和标注按文件名（`<病例>_<颌位>`）配对，不要求固定的目录层级。

拿到数据后计划做的事：
1. Teeth3DS 测试集上：朝向归一化（`--check-orientation`）、编号纠错统计、缺牙生成与真牙对比（目前只在 1 个病例上做过）；
2. 用训练集构建标准牙模板库，替换内置的简化模板；
3. Poseidon3D：先看格式，写读取适配，再在困难病例上评估。

已在本环境验证：`fetch` 的下载、解压、按文件名配对、评估整条链路（用本地 zip 模拟）；gsutil 通过代理访问 GCS 公开桶正常。私有桶 + 服务账号这一步要等你配好才能实测。
