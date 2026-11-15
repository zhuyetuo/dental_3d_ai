#!/bin/bash
# TeethSplineSeg 在“原样 + 模拟缺牙 + 缺隙闭合”数据上微调（从作者权重开始，30 轮）。等今晚的评分卡和数据准备都完成再占 GPU。
cd ~/zyt/dental_3d_ai
SS=third_party/TeethSplineSeg
python3 - <<'EOF'
from pathlib import Path
p = Path("third_party/TeethSplineSeg/train_network.py")
s = p.read_text()
if "--init_weights" not in s:
    s = s.replace('    parser.add_argument("--ckpt", default=None, help="Optional checkpoint for resuming training.")\n',
                  '    parser.add_argument("--ckpt", default=None, help="Optional checkpoint for resuming training.")\n'
                  '    parser.add_argument("--init_weights", default=None, help="只加载这个检查点的权重，从第 1 轮开始训练（微调用）")\n', 1)
    s = s.replace("    model = build_model(args)\n",
                  "    model = build_model(args)\n"
                  "    if args.init_weights:\n"
                  "        state = torch.load(args.init_weights, map_location=\"cpu\", weights_only=False)[\"state_dict\"]\n"
                  "        missing, unexpected = model.load_state_dict(state, strict=False)\n"
                  "        print(f\"init_weights: missing {len(missing)}, unexpected {len(unexpected)}\", flush=True)\n", 1)
    p.write_text(s)
    print("patched train_network (--init_weights)")
EOF
while ! grep -q AUG_PREP_DONE runs/ss_aug_prep.log 2>/dev/null; do sleep 60; done
while ! grep -q MORE_CARDS_DONE runs/rd_more.log 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M)] 开始微调"
cd $SS
../../.venv-splineseg/bin/python train_network.py --root_folder data/aug --experiment_name aug --experiment_version ft30 \
  --epochs 30 --devices 0 --num_workers 16 --init_weights "$(realpath ../../models/teethsplineseg/best.ckpt)" 2>&1 \
  | tr '\r' '\n' | grep -v 'it/s\|s/it' > ../../runs/ss_ft.log
echo "[$(date +%H:%M)] SS_FT_DONE" >> ../../runs/ss_ft.log
