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

## TGNet

1. 从 TGNet 作者的 Google Drive 下载 `ckpts(new).zip`：
   https://drive.google.com/drive/folders/15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby
2. 解压，把这两个文件放到 `models/tgnet/`：

```
models/tgnet/tgnet_fps.h5
models/tgnet/tgnet_bdl.h5
```

注意：TGNet 仓库没有 LICENSE 文件，权重仅供研究评估，商用前需要联系作者取得授权。
