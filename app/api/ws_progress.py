"""
WebSocket real-time pipeline progress streaming.

Endpoint: WS /upload/progress/{job_id}

The client connects immediately after POST /upload/transcribe and receives
a stream of JSON events as the pipeline runs:

  {"type": "transcribing", "progress": 45, "message": "Transcribing audio…", "ts": 1234567890.0}
  {"type": "summarizing",  "progress": 75, "message": "Extracting insights…", "ts": 1234567891.0}
  {"type": "done",         "progress": 100, "message": "Processing complete", "ts": 1234567892.0}

The WebSocket closes automatically after DONE or FAILED is received.

The client can also send:
  {"action": "cancel"}   — reserved for future use (pipeline cancellation)
  {"action": "ping"}     — keepalive; server replies {"type": "pong"}
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.pipeline.event_bus import stream_job_events
from app.db.redis_client import RedisJobStore, get_redis

log = logging.getLogger("ws_progress")
router = APIRouter(tags=["progress"])

# Maximum time (seconds) to wait for first event before checking if job exists
_INITIAL_WAIT = 30.0
# Keepalive ping interval (seconds)
_PING_INTERVAL = 15.0


@router.websocket("/upload/progress/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str):
    """Stream pipeline progress events for job_id in real-time."""
    await websocket.accept()
    log.info("[ws_progress] client connected job=%s", job_id)

    # Verify job exists before streaming
    r = await get_redis()
    store = RedisJobStore(r)
    if not await store.exists(job_id):
        await websocket.send_text(json.dumps({
            "type": "error", "message": "Job not found", "job_id": job_id
        }))
        await websocket.close(code=4004)
        return

    # Start keepalive ping task
    ping_task = asyncio.ensure_future(_keepalive(websocket))

    try:
        async for event in stream_job_events(job_id):
            try:
                # Forward event to client — only keep the fields the UI needs
                payload = {
                    "type":     event.get("type", ""),
                    "stage":    event.get("stage", ""),
                    "progress": int(event.get("progress", 0)),
                    "message":  event.get("message", ""),
                    "ts":       float(event.get("ts", 0)),
                }
                await websocket.send_text(json.dumps(payload))

                # Handle client messages (non-blocking)
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                    msg = json.loads(raw)
                    if msg.get("action") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                except (asyncio.TimeoutError, json.JSONDecodeError):
                    pass

                # Stop streaming on terminal events
                if payload["type"] in ("done", "failed"):
                    break

            except WebSocketDisconnect:
                log.info("[ws_progress] client disconnected job=%s", job_id)
                break
            except Exception as exc:
                log.warning("[ws_progress] send error job=%s: %s", job_id, exc)
                break

    except WebSocketDisconnect:
        log.info("[ws_progress] client disconnected early job=%s", job_id)
    except Exception as exc:
        log.error("[ws_progress] stream error job=%s: %s", job_id, exc)
    finally:
        ping_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass
        log.info("[ws_progress] connection closed job=%s", job_id)


async def _keepalive(ws: WebSocket) -> None:
    """Send periodic pings to keep the connection alive through proxies."""
    try:
        while True:
            await asyncio.sleep(_PING_INTERVAL)
            await ws.send_text(json.dumps({"type": "heartbeat"}))
    except Exception:
        pass
