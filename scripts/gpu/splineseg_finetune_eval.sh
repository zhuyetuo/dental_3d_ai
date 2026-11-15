#!/bin/bash
# 微调结束后：用新权重在公平测试集、Bite2Text 真实缺牙上各测一遍（和微调前同样的流程）
cd ~/zyt/dental_3d_ai && . .venv/bin/activate
while ! grep -q SS_FT_DONE runs/ss_ft.log 2>/dev/null; do sleep 60; done
SS=third_party/TeethSplineSeg
PY=$PWD/.venv-splineseg/bin/python
CK=$(ls -t $SS/logs/aug/ft30/*.ckpt 2>/dev/null | grep -i -m1 'tir\|best') || true
[ -n "$CK" ] || CK=$(ls -t $SS/logs/aug/ft30/*.ckpt | head -1)
echo "用的权重：$CK"
mkdir -p runs/splineseg_ft
# 公平测试集
(cd $SS && $PY inference.py --ckpt "$(realpath ../../$CK 2>/dev/null || realpath $CK)" --devices 0 --root_folder "$(realpath .)/data/fair" \
   --output_dir "$(realpath ../..)/runs/splineseg_ft/fair" --overwrite --upsample_workers 24 2>&1 | tr '\r' '\n' | grep -a -A12 '指标\|score' | tail -n 14)
# 真实缺牙
(cd $SS && $PY inference.py --ckpt "$(realpath $CK)" --devices 0 --root_folder "$(realpath .)/data/bite" \
   --output_dir "$(realpath ../..)/runs/splineseg_ft/bite_infer" --overwrite --upsample_workers 24 > /dev/null 2>&1)
mkdir -p runs/splineseg_ft/bite
ln -sfn "$(realpath data/bite_splineseg)" runs/splineseg_ft/bite/mesh
cd ml
python -m dental_ai.external import --scheme splineseg --pred-dir ../runs/splineseg_ft/bite_infer/postprocessed --data-dir ../data/bite_splineseg --out ../runs/splineseg_ft/bite/pred | tail -1
python -m dental_ai.eval_real_missing run --zip ~/zyt/dental_data/Bite2Text.zip --truth ../runs/real_missing/truth.json --model none \
  --pred-dir ../runs/splineseg_ft/bite/pred --out ../runs/splineseg_ft/bite/real_missing --workers 24 2>&1 | grep -A4 '"confirmed_missing"'
echo SS_FT_EVAL_DONE
