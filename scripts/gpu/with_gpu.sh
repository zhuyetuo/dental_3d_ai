#!/bin/bash
# 要用 GPU 的任务包一层：先让空闲训练（idle_trainer.sh）存好检查点退出，再跑；跑完撤掉标记，空闲训练自己接着训。
#   bash scripts/gpu/with_gpu.sh <命令...>
# 标记 runs/.idle_pause.<本进程 pid>（绝对路径）；就算这里异常退出没删掉，守护进程也只认 pid 还活着的标记。
REPO=$HOME/zyt/dental_3d_ai
flag=$REPO/runs/.idle_pause.$$
touch "$flag"
trap 'rm -f "'"$flag"'"' EXIT
for _ in $(seq 1 120); do
  [ -f $REPO/runs/.idle_trainer.pid ] && kill -0 "$(cat $REPO/runs/.idle_trainer.pid)" 2>/dev/null || break
  sleep 5
done
"$@"
