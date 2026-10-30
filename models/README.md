# 模型权重

权重文件不提交到 git（见 .gitignore），按下面的目录放好后由 docker-compose 挂载进容器。

## TGNet

1. 从 TGNet 作者的 Google Drive 下载 `ckpts(new).zip`：
   https://drive.google.com/drive/folders/15oP0CZM_O_-Bir18VbSM8wRUEzoyLXby
2. 解压，把这两个文件放到 `models/tgnet/`：

```
models/tgnet/tgnet_fps.h5
models/tgnet/tgnet_bdl.h5
```

注意：TGNet 仓库没有 LICENSE 文件，权重仅供研究评估，商用前需要联系作者取得授权。
