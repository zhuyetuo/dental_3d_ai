"""ToothGroupNetwork（TGNet，3DTeethSeg'22 第一名）适配器。

TGNet 依赖 PyTorch 1.7.1 + CUDA 11.0 和自定义的 pointops 算子，与本项目的环境差别很大，
所以不直接 import，而是在单独的环境（推荐用它的 Docker 镜像）里通过命令行调用：

    python start_inference.py --input_dir_path <dir> --split_txt_path <txt> --save_path <out>
        --model_name tgnet_fps --checkpoint_path <fps.h5> --checkpoint_path_bdl <bdl.h5>

输入要求：<case>_<jaw>.obj，Y 轴朝后；输出：<case>_<jaw>.json，格式同 Teeth3DS。
注意：TGNet 仓库没有 LICENSE 文件，商用前需要联系作者确认授权。
"""

from __future__ import annotations

import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from ..fdi import Jaw
from ..io import load_labels
from .base import Segmenter


@dataclass
class TGNetConfig:
    repo_dir: Path  # ToothGroupNetwork 仓库路径
    checkpoint_fps: Path
    checkpoint_bdl: Path
    python: str = "python"  # TGNet 环境中的解释器，如 "conda run -n tgnet python"


class TGNetSegmenter(Segmenter):
    name = "tgnet"

    def __init__(self, config: TGNetConfig):
        self.config = config

    def predict(self, mesh: trimesh.Trimesh, jaw: Jaw) -> np.ndarray:
        cfg = self.config
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            case = "case"
            (tmp / "in" / case).mkdir(parents=True)
            mesh.export(tmp / "in" / case / f"{case}_{jaw}.obj")
            split = tmp / "split.txt"
            split.write_text(f"{case}\n")
            out = tmp / "out"
            out.mkdir()
            cmd = [
                *shlex.split(cfg.python),
                "start_inference.py",
                "--input_dir_path", str(tmp / "in"),
                "--split_txt_path", str(split),
                "--save_path", str(out),
                "--model_name", "tgnet_fps",
                "--checkpoint_path", str(cfg.checkpoint_fps),
                "--checkpoint_path_bdl", str(cfg.checkpoint_bdl),
            ]  # fmt: skip
            subprocess.run(cmd, cwd=cfg.repo_dir, check=True)
            labels, _ = load_labels(out / f"{case}_{jaw}.json")
        return labels
