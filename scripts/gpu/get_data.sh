#!/bin/bash
# 下载数据集并导出。需要 .env 里的 GOOGLE_API_KEY（Drive API 密钥）和 DRIVE_FOLDER_ID（共享文件夹 ID）。
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; [ -f .env ] && . ./.env; set +a
: "${DRIVE_FOLDER_ID:?在 .env 里设置 DRIVE_FOLDER_ID（Google Drive 共享文件夹链接里 folders/ 后面那一段）}"
python scripts/download_datasets.py fetch --name drive --src "gdrive://${DRIVE_FOLDER_ID}" --no-extract
Z=data/drive/raw/Teeth3DS
SPLIT=$Z/Teeth3DS+
cd ml
# Teeth3DS 官方测试集（随机 100 个，与 CPU 基线评估用的同一批）和挑战赛私有测试集（评估 TGNet 用，避免它见过的训练数据）
python - <<'PY'
import random
from pathlib import Path
base = Path("../data/drive/raw/Teeth3DS/Teeth3DS+")
test = [l.strip() for f in ("testing_lower.txt", "testing_upper.txt") for l in open(base / "Teeth3DS_train_test_split" / f) if l.strip()]
random.seed(0); random.shuffle(test)
Path("/tmp/t3d_test100.txt").write_text("\n".join(test[:100]))
private = [l.strip() for l in open(base / "3DTeethSeg22_challenge_train_test_split" / "private-testing-set.txt") if l.strip()]
random.seed(0); random.shuffle(private)
Path("/tmp/challenge_test100.txt").write_text("\n".join(private[:100]))
PY
python -m dental_ai.datasets teeth3ds --zips ../$Z --split /tmp/t3d_test100.txt --out ../data/teeth3ds_test100
python -m dental_ai.datasets teeth3ds --zips ../$Z --split /tmp/challenge_test100.txt --out ../data/challenge_test100
P=../data/drive/raw/Poseidon3D/poseidon3d.zip
for s in train test-simple test-complex; do python -m dental_ai.datasets poseidon3d --zip $P --split $s --out ../data/poseidon3d/$s; done
echo "数据就绪：data/teeth3ds_test100、data/challenge_test100、data/poseidon3d/{train,test-simple,test-complex}"
