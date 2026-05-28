"""
Redis Streams event bus for the meeting processing pipeline.

Each pipeline job publishes typed events to a job-scoped stream:
  stream:pipeline:<job_id>

A global jobs stream broadcasts all job lifecycle events for monitoring:
  stream:jobs

Event schema (all values stored as strings in the Redis hash):
  type      — PipelineEvent enum value
  job_id    — UUID string
  stage     — current stage name
  progress  — 0-100
  message   — human-readable status
  payload   — optional JSON-encoded extra data
  ts        — Unix timestamp float

Consumer groups allow multiple worker replicas to share load without
double-processing. A single XADD + XREADGROUP flow provides at-least-once
delivery with ACK.
"""

from __future__ import annotations

import json
import logging
import time
from enum import Enum
from typing import Any, AsyncIterator

import redis.asyncio as aioredis

from app.config import (
    REDIS_STREAM_PIPELINE,
    REDIS_STREAM_JOBS,
    REDIS_JOB_TTL,
    PIPELINE_STREAM_MAXLEN,
    PIPELINE_CONSUMER_GROUP,
    PIPELINE_CONSUMER_NAME,
    PIPELINE_BLOCK_MS,
)
from app.db.redis_client import get_redis

log = logging.getLogger("event_bus")


class PipelineEvent(str, Enum):
    QUEUED        = "queued"
    UPLOADING     = "uploading"
    AUDIO_EXTRACT = "extracting_audio"
    TRANSCRIBING  = "transcribing"
    SUMMARIZING   = "summarizing"
    RAG_INDEXING  = "rag_indexing"
    DONE          = "done"
    FAILED        = "failed"


async def _ensure_group(r: aioredis.Redis, stream: str, group: str) -> None:
    """Create consumer group if it does not already exist (MKSTREAM flag creates the stream too)."""
    try:
        await r.xgroup_create(stream, group, id="0", mkstream=True)
    except aioredis.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def publish_event(
    job_id: str,
    event: PipelineEvent,
    *,
    progress: int = 0,
    message: str = "",
    payload: dict[str, Any] | None = None,
) -> None:
    """
    Publish a pipeline event to both the job-scoped stream and the global jobs stream.
    Non-blocking — failures are logged but never propagate to callers.
    """
    try:
        r = await get_redis()
        record: dict[str, str] = {
            "type":     event.value,
            "job_id":   job_id,
            "stage":    event.value,
            "progress": str(progress),
            "message":  message,
            "payload":  json.dumps(payload or {}),
            "ts":       str(time.time()),
        }
        job_stream = f"{REDIS_STREAM_PIPELINE}:{job_id}"

        await r.xadd(job_stream, record, maxlen=PIPELINE_STREAM_MAXLEN, approximate=True)
        await r.expire(job_stream, REDIS_JOB_TTL)

        # Global monitoring stream — lightweight (only type + job_id + progress)
        await r.xadd(
            REDIS_STREAM_JOBS,
            {"type": event.value, "job_id": job_id, "progress": str(progress), "ts": record["ts"]},
            maxlen=PIPELINE_STREAM_MAXLEN,
            approximate=True,
        )
    except Exception as exc:
        log.warning("[event_bus] publish failed job=%s event=%s: %s", job_id, event, exc)


async def read_job_events(
    job_id: str,
    *,
    last_id: str = "0",
) -> list[dict[str, Any]]:
    """
    Read all events for a job from last_id onwards (used by WebSocket progress handler).
    Returns a list of dicts with normalised fields.
    """
    try:
        r = await get_redis()
        stream = f"{REDIS_STREAM_PIPELINE}:{job_id}"
        results = await r.xread({stream: last_id}, count=50)
        events = []
        for _stream, messages in results:
            for msg_id, fields in messages:
                events.append({"id": msg_id, **fields})
        return events
    except Exception as exc:
        log.warning("[event_bus] read failed job=%s: %s", job_id, exc)
        return []


async def stream_job_events(
    job_id: str,
) -> AsyncIterator[dict[str, Any]]:
    """
    Async generator: yield new events for job_id as they arrive.
    Yields immediately for existing events, then blocks waiting for new ones.
    Stops when a DONE or FAILED event is seen.
    """
    r = await get_redis()
    stream = f"{REDIS_STREAM_PIPELINE}:{job_id}"
    last_id = "0"  # start from the beginning

    while True:
        try:
            results = await r.xread({stream: last_id}, count=20, block=PIPELINE_BLOCK_MS)
        except Exception as exc:
            log.warning("[event_bus] stream read error job=%s: %s", job_id, exc)
            break

        if not results:
            # Timeout — check if job is still alive or has been cleaned up
            if not await r.exists(stream):
                break
            continue

        for _stream, messages in results:
            for msg_id, fields in messages:
                last_id = msg_id
                event = {"id": msg_id, **fields}
                yield event
                if fields.get("type") in (PipelineEvent.DONE.value, PipelineEvent.FAILED.value):
                    return


async def bootstrap_consumer_group(job_id: str) -> None:
    """Ensure the consumer group exists for a job stream (call before workers start consuming)."""
    r = await get_redis()
    stream = f"{REDIS_STREAM_PIPELINE}:{job_id}"
    await _ensure_group(r, stream, PIPELINE_CONSUMER_GROUP)
