#!/bin/bash
# 空闲时训练的守护进程：GPU 没有别的任务时跑长期训练（能接着训），别的任务要用 GPU 时先让训练存好检查点、退出让路。
#
#   nohup setsid bash scripts/gpu/idle_trainer.sh > runs/idle_trainer.log 2>&1 &
#
# “别的任务”= 显卡上不属于下面这些常驻进程的计算进程：
#   本守护进程起的训练、网页的 GPU worker（docker 里的 rq worker / uvicorn）、TeethSplineSeg 分割服务、nvitop
# 别的任务出现 → 给训练发 SIGTERM（它把进度存进检查点再退出）；GPU 连续 3 分钟没有别的任务 → 接着训。
# 训练任务按 runs/idle_jobs.txt 一行一个（第一个没做完的就是当前任务），每行：名字|命令（命令里用 --forever，out 存在就接着训）。
# 某个任务想停掉：在 runs/idle_jobs.txt 里删掉那一行或行首加 #；全部暂停：touch runs/.idle_pause
cd ~/zyt/dental_3d_ai
JOBS=runs/idle_jobs.txt
PIDF=runs/.idle_trainer.pid
echo $$ > runs/.idle_daemon.pid
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }

others_on_gpu() {  # 显卡上“别的任务”的进程数
  local mine=""
  [ -f $PIDF ] && mine=$(cat $PIDF)
  local n=0
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    [ -n "$mine" ] && { [ "$pid" = "$mine" ] || pstree -p "$mine" 2>/dev/null | grep -q "($pid)"; } && continue
    cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null)
    case "$cmd" in
      *"rq worker"*|*uvicorn*|*services/splineseg/server.py*|*nvitop*|"") continue ;;
    esac
    n=$((n + 1))
  done
  echo $n
}

current_job() { grep -v '^\s*#' $JOBS 2>/dev/null | grep -v '^\s*$' | head -1; }

trainer_alive() { [ -f $PIDF ] && kill -0 "$(cat $PIDF)" 2>/dev/null; }

free_since=0
while true; do
  job=$(current_job)
  # with_gpu.sh 要用 GPU 时会放 runs/.idle_pause.<pid>：只认那个 pid 还活着的（异常退出留下的过期标记删掉）；runs/.idle_pause 是手动全暂停
  live_pause=0
  for f in runs/.idle_pause.*; do
    [ -e "$f" ] || continue
    if kill -0 "${f##*.}" 2>/dev/null; then live_pause=1; else rm -f "$f"; fi
  done
  [ -e runs/.idle_pause ] && live_pause=1
  if [ -z "$job" ] || [ $live_pause = 1 ]; then
    if trainer_alive; then
      log "暂停（有任务要用 GPU）/ 没有训练任务：让训练存好检查点退出"
      kill -TERM "$(cat $PIDF)"
      for _ in $(seq 1 60); do trainer_alive || break; sleep 5; done
      trainer_alive && kill -9 "$(cat $PIDF)"
      rm -f $PIDF
    fi
    free_since=0; sleep 20; continue
  fi
  name=${job%%|*}; cmd=${job#*|}
  busy=$(others_on_gpu)
  if trainer_alive; then
    if [ "$busy" -gt 0 ]; then
      log "有别的 GPU 任务（$busy 个）：让 $name 存好检查点退出"
      kill -TERM "$(cat $PIDF)"
      for _ in $(seq 1 60); do trainer_alive || break; sleep 5; done
      trainer_alive && { log "5 分钟没退出，强制结束"; kill -9 "$(cat $PIDF)"; }
      rm -f $PIDF; free_since=0
    fi
  else
    if [ "$busy" -eq 0 ]; then
      [ $free_since -eq 0 ] && free_since=$(date +%s)
      if [ $(( $(date +%s) - free_since )) -ge 180 ]; then
        log "GPU 空了 3 分钟：开始 / 接着训 $name"
        mkdir -p runs/idle/$name
        # exec：记下来的 pid 就是训练进程本身，SIGTERM 直接发给它（命令里不要用管道）
        (cd ml && { setsid bash -c "exec $cmd" >> ../runs/idle/$name/train.log 2>&1 < /dev/null & echo $! > ../$PIDF; })
        free_since=0
      fi
    else
      free_since=0
    fi
  fi
  sleep 60
done
