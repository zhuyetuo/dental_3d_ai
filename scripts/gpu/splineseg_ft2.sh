#!/bin/bash
# TeethSplineSeg 第二轮微调（11-24）：追加闭合缺隙为主的模拟数据 → 预处理 → 从现用权重接着训（固定学习率 1e-4）→ 真实缺牙单独评测
# 预处理的瓶颈是最远点采样：GPU 进程挤一张卡、CPU 进程各算各的，两边一起跑（按速度 3:1 分样本）
cd ~/zyt/dental_3d_ai && . .venv/bin/activate
LOG=runs/ss_ft2.log
[ -n "$SKIP_EXPORT" ] || { echo "[$(date +%H:%M)] 导出追加数据" >> $LOG; python scripts/gpu/splineseg_aug2_export.py --workers 48 2>&1 | tail -3 >> $LOG; }
SRC=$(realpath data/splineseg_aug)
SS=$(realpath third_party/TeethSplineSeg)
ROOT=$SS/data/aug2
OLD=$SS/data/aug
CACHE=processed_32000_points_7dim_arch_gsn_jawtype
PY=$PWD/.venv-splineseg/bin/python
mkdir -p $ROOT
ln -sfn "$OLD/$CACHE" "$ROOT/$CACHE"   # 同一个预处理目录（新样本名字都不重复）
ln -sfn "$SRC" "$ROOT/raw"
G=12; C=36; W=/tmp/ssaug2
for d in $W/root*/$CACHE; do cp -n $d/data_*.pt "$OLD/$CACHE"/ 2>/dev/null; done
rm -rf $W; mkdir -p $W
echo "[$(date +%H:%M)] 预处理（GPU $G 份 + CPU $C 份）：$(python scripts/gpu/ss_prep_mixed.py $SRC $OLD/$CACHE $G $C)" >> $LOG
for i in $(seq 0 $((G + C - 1))); do
  if [ $i -lt $G ]; then DEV=""; else DEV="CUDA_VISIBLE_DEVICES="; fi
  (cd $SS && env $DEV OMP_NUM_THREADS=1 $PY prepare_dataset.py --source $W/src$i --root $W/root$i > $W/log$i 2>&1; cp $W/root$i/$CACHE/data_*.pt "$OLD/$CACHE"/ 2>/dev/null || true) &
done
wait
for jaw in lower upper; do
  cat "$OLD/training_$jaw.txt" "$SRC/training2_$jaw.txt" | grep . | sort -u | while read n; do [ -f "$OLD/$CACHE/data_$n.pt" ] && echo "$n"; done > "$ROOT/training_$jaw.txt"
  cp "$OLD/testing_$jaw.txt" "$ROOT/"
done
echo "[$(date +%H:%M)] 训练样本 $(cat $ROOT/training_*.txt | wc -l)（第一轮 $(cat $OLD/training_*.txt | grep -c .)）" >> $LOG
grep -h -i "failed\|error" $W/log* | head -5 >> $LOG
echo "[$(date +%H:%M)] 开始训练" >> $LOG
INIT=$(realpath models/teethsplineseg/finetuned_aug.ckpt)
(cd $SS && TSS_LR=1e-4 TSS_CONST_LR=1 $PY train_network.py --root_folder data/aug2 --experiment_name aug --experiment_version ft2 \
  --epochs 4 --devices 0 --num_workers 16 --init_weights "$INIT" 2>&1 | tr '\r' '\n' | grep -v 'it/s\|s/it' > $(realpath ../..)/runs/ss_ft2_train.log)
echo "[$(date +%H:%M)] 训练完" >> $LOG
ls $SS/logs/aug/ft2/ >> $LOG
# 真实缺牙单独评测：最好的 t_miou 和最后一轮
cd ml
for CK in $(ls -t $SS/logs/aug/ft2/best-val_t_miou-*.ckpt | head -1) $SS/logs/aug/ft2/latest-*.ckpt; do
  tag=$(basename $CK .ckpt | tr '=' '_')
  O=$(realpath -m ../runs/splineseg_ft2/$tag)
  mkdir -p $O; ln -sfn "$(realpath ../data/bite_splineseg)" $O/mesh   # 评测按 pred 旁边的 mesh 找口扫
  (cd $SS && $PY inference.py --ckpt "$CK" --devices 0 --root_folder "$SS/data/bite" --output_dir "$(realpath $O)/infer" --overwrite --upsample_workers 24 > /dev/null 2>&1)
  python -m dental_ai.external import --scheme splineseg --pred-dir $O/infer/postprocessed --data-dir ../data/bite_splineseg --out $O/pred | tail -1
  python -m dental_ai.eval_real_missing run --zip ~/zyt/dental_data/Bite2Text.zip --truth ../runs/real_missing/truth.json --model none \
    --pred-dir $O/pred --out $O/real_missing --workers 24 > $O/eval.log 2>&1
  echo "$tag" >> ../$LOG
  python3 -c "import json;s=json.load(open('$O/real_missing/summary.json'));print({k:s[k] for k in ('confirmed_missing',) if k in s})" >> ../$LOG
done
echo "[$(date +%H:%M)] SS_FT2_DONE" >> ../$LOG
