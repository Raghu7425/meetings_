"""
Structured JSON logging with correlation IDs.

Replaces the plain-text formatter in main.py. Every log record emits a JSON
object so log aggregators (Loki, Datadog, CloudWatch) can index all fields:

  {"ts": "2026-05-28T10:00:00Z", "level": "INFO", "logger": "upload",
   "msg": "pipeline started", "job_id": "abc123", "stage": "transcribing",
   "duration_ms": 420}

Usage:
  from app.utils.logging_config import configure_logging
  configure_logging()          # call once at app startup

Injecting context:
  from app.utils.logging_config import bind_context, clear_context
  bind_context(job_id="abc123", stage="transcribing")
  log.info("pipeline started")   # job_id + stage auto-appended
  clear_context()
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from contextvars import ContextVar
from logging.handlers import RotatingFileHandler
from typing import Any

from app.config import LOG_FILE, LOG_MAX_BYTES, LOG_BACKUP_COUNT, LOGS_DIR

import os
os.makedirs(LOGS_DIR, exist_ok=True)

# Thread-local context injected into every log record in the same async task
_log_context: ContextVar[dict[str, Any]] = ContextVar("log_context", default={})


def bind_context(**kwargs: Any) -> None:
    """Attach key-value pairs that appear on every subsequent log call in this coroutine."""
    ctx = dict(_log_context.get())
    ctx.update(kwargs)
    _log_context.set(ctx)


def clear_context() -> None:
    _log_context.set({})


class _JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    SKIP = frozenset({"name", "msg", "args", "created", "relativeCreated",
                      "thread", "threadName", "processName", "process",
                      "msecs", "pathname", "filename", "module", "lineno",
                      "funcName", "stack_info", "exc_info", "exc_text"})

    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, Any] = {
            "ts":     self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }

        # Merge in async context (job_id, session_id, stage, etc.)
        doc.update(_log_context.get())

        # Extra fields attached via log.info(..., extra={...})
        for k, v in record.__dict__.items():
            if k not in self.SKIP and not k.startswith("_"):
                doc[k] = v

        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)

        return json.dumps(doc, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """
    Install structured JSON handlers on the root logger.
    Safe to call multiple times — duplicate handlers are not added.
    """
    root = logging.getLogger()
    if any(isinstance(h, _JSONFormatter) for h in
           [getattr(h, "formatter", None) for h in root.handlers]):
        return  # already configured

    fmt = _JSONFormatter()

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    root.setLevel(level)
    # Remove any existing plain handlers added before this call
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    # Quiet down noisy libraries
    for noisy in ("speechbrain", "httpx", "httpcore", "urllib3",
                  "asyncio", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
