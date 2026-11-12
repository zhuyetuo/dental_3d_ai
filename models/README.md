# 模型权重

权重文件不提交到 git（见 .gitignore），按下面的目录放好后由 docker-compose 挂载进容器。

## 标准牙模板（templates）

`models/templates/` 下放 `<jaw>_<position>.stl`（如 `lower_6.stl`，同时用于 36 和 46），牙齿局部坐标系：x 近中、y 颊侧、z 咬合，原点在牙颈线中心，底边是牙颈线。目录为空时使用内置参数化模板。

从 Teeth3DS 格式的标注数据构建（每个牙位取近远中宽度最接近中位数的一颗牙冠，并写 `index.json` 记录来源）：

```bash
cd ml && python -m dental_ai.templates build --data-dir <obj 目录> --gt-dir <标注目录> --out ../models/templates
```

导出内置模板查看：`python -m dental_ai.templates export-builtin --out <目录>`。

许可：用 Teeth3DS（CC BY-NC-ND 4.0）构建的模板只能用于研究；商用请用自有授权数据构建。

## 自己训练的模型

| 文件 | 训练脚本 | 用法 |
|---|---|---|
| `models/segmentation/deep.pt` | `scripts/gpu/train_deep.sh` | 后端环境变量 `DEEP_MODEL`。DGCNN + 全局自注意力（`dgcnn_attn`），训练数据里加了模拟缺牙 |
| `models/segmentation/deep_dgcnn.pt` | `python -m dental_ai.segmentation.deep train --arch dgcnn` | 原来的 DGCNN 结构、同样的数据，文件小（1.6 MB）、推理快，各项指标略低；`DEEP_MODEL` 指向它即可换用 |
| `models/generation/toothgen_open.pt` | `scripts/gpu/train_toothgen.sh` | `pipeline --gen-model` 或后端环境变量 `GEN_MODEL` |
| `models/margin/margin.pt` | `scripts/gpu/train_margin.sh` | 还没接入流程，`python -m dental_ai.margin.learned eval` |

这三个文件随仓库提交（.gitignore 里单独放行），`docker compose up --build` 后后端直接使用前两个。
分割和生成模型用 Teeth3DS（CC BY-NC-ND 4.0）训练，**只能用于非商业研究**，商用需要用自有授权数据重新训练；颈缘线模型的训练数据是 MIT 许可。

## TGNet

1. 从 TGNet 作者的 Google Drive 下载 `ckpts(new).zip`：
   https://drive.google.com/drive/folders/15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby
2. 解压，把这两个文件放到 `models/tgnet/`：

```
models/tgnet/tgnet_fps.h5
models/tgnet/tgnet_bdl.h5
```

注意：TGNet 仓库没有 LICENSE 文件，权重仅供研究评估，商用前需要联系作者取得授权。
