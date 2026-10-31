"""处理任务的调度。

- 设置了 REDIS_URL：放进 RQ 队列，由单独的 worker 进程处理（docker-compose 的 worker 服务，可以开多个），
  API 进程不被重计算拖慢；病例文件通过共享的卷（DENTAL_DATA_DIR）在 API 和 worker 之间共用；
- 没有设置：在 API 进程里用 FastAPI 的后台任务处理（本地开发、单机演示）。
"""

from __future__ import annotations

import os

from fastapi import BackgroundTasks

from .worker import process_case

QUEUE_NAME = "dental"
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "1800"))  # 秒，大网格 + 多颗牙冠也够


def _queue():
    url = os.environ.get("REDIS_URL")
    if not url:
        return None
    from redis import Redis
    from rq import Queue

    return Queue(QUEUE_NAME, connection=Redis.from_url(url))


def submit(case_id: str, background: BackgroundTasks) -> str:
    """提交处理任务，返回调度方式（"queue" 或 "in_process"）。"""
    queue = _queue()
    if queue is None:
        background.add_task(process_case, case_id)
        return "in_process"
    queue.enqueue(process_case, case_id, job_timeout=JOB_TIMEOUT, job_id=f"case-{case_id}-{os.urandom(4).hex()}")
    return "queue"
