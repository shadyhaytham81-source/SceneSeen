"""Minimal in-process job runner: one worker thread (analysis is CPU/GPU heavy),
job state kept in memory and polled by the UI."""
from __future__ import annotations

import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Callable

log = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"      # queued | running | done | error
    stage: str = ""
    stage_label: str = ""
    progress: float = 0.0       # 0..1 overall
    error: str = ""
    result: dict | None = None
    started: float = field(default_factory=time.time)

    def public(self) -> dict:
        d = asdict(self)
        d.pop("result", None)
        return d


class JobRunner:
    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sceneseen-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def submit(self, job_id: str, kind: str, fn: Callable[[Job], dict]) -> Job:
        with self._lock:
            existing = self._jobs.get(job_id)
            if existing and existing.status in ("queued", "running"):
                return existing
            job = Job(job_id, kind)
            self._jobs[job_id] = job

        def run():
            job.status = "running"
            try:
                job.result = fn(job)
                job.progress, job.status = 1.0, "done"
            except Exception as e:  # surfaced to the UI, full trace to the log
                log.error("job %s failed:\n%s", job_id, traceback.format_exc())
                job.status, job.error = "error", _friendly(e)

        self._pool.submit(run)
        return job


def _friendly(e: Exception) -> str:
    from sceneseen.media import VideoError

    if isinstance(e, VideoError):
        return str(e)
    return f"Something went wrong while processing this video ({type(e).__name__}: {e})"
