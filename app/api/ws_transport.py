"""
Handles WebSocket message sending and control actions.

- Safely sends JSON and audio data to client without crashing
- Maintains connection using ping/pong (keepalive)

- Processes control messages from client:
  • interrupt → stop current AI response
  • ping/pong → connection check
  • set_voice → change TTS voice
  • reset → clear session and start fresh

- Updates session state based on control actions

This file manages communication safety and control between client and server.
"""


import json
import asyncio
import logging
from fastapi import WebSocket, WebSocketDisconnect
from app.core.voices import is_supported_voice
from app.core.agent import clear_session, get_session_state, update_session_voice
from app.api.ws_session import SessionState, _reset_current_utterance_tracking, _reset_interrupt_tracking
from app.config import WS_PING_INTERVAL, WS_PING_MAX_MISSED, WS_PING_TIMEOUT


log = logging.getLogger("websocket")


async def _safe_send_json(ws: WebSocket, data: dict):
    try:
        await ws.send_json(data)
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as e:
        log.debug(f"[WS] _safe_send_json error: {e}")



async def _safe_send_bytes(ws: WebSocket, data: bytes):
    try:
        await ws.send_bytes(data)
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception as e:
        log.debug(f"[WS] _safe_send_bytes error: {e}")



async def _keepalive(ws: WebSocket, s: SessionState):

    missed = 0
    try:
        while True:
            try:
                await asyncio.sleep(WS_PING_INTERVAL)
            except asyncio.CancelledError:
                return

            s.pong_received.clear()
            await _safe_send_json(ws, {"type": "server_ping"})

            try:
                await asyncio.wait_for(
                    asyncio.shield(s.pong_received.wait()),
                    timeout=WS_PING_TIMEOUT,
                )
                missed = 0
            except asyncio.TimeoutError:
                missed += 1
                log.warning(f"[{s.id}] ping missed ({missed}/{WS_PING_MAX_MISSED})")
                if missed >= WS_PING_MAX_MISSED:
                    log.error(f"[{s.id}] keepalive timeout — closing")
                    try:
                        await ws.close(code=1001, reason="keepalive timeout")
                    except Exception:
                        pass
                    return
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.error(f"[{s.id}] Keepalive error: {e}")
                return

    except asyncio.CancelledError:
        pass

    except Exception as e:
        log.error(f"[{s.id}] _keepalive unexpected error: {e}")



async def _handle_control(ws: WebSocket, s: SessionState, raw: str):
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning(f"[{s.id}] Malformed control JSON: {e}")
        return None

    t = msg.get("type")

    try:
        if t == "interrupt":
            log.info(f"[{s.id}] client requested interrupt")
            return "interrupt"

        if t == "ping":
            await _safe_send_json(ws, {"type": "pong", "session": get_session_state(s.id)})
            return None

        if t == "pong":
            s.pong_received.set()
            return None

        if t == "set_voice":
            voice = msg.get("voice", "")
            if isinstance(voice, str):
                voice = voice.strip()

            if not is_supported_voice(voice):
                await _safe_send_json(ws, {"type": "error", "text": "Invalid voice"})
                return None

            update_session_voice(s.id, voice)
            await _safe_send_json(ws, {"type": "voice_changed", "voice": voice})
            log.info(f"[{s.id}] voice changed to: {voice}")
            return None

        if t == "reset":
            s.cancel_pipeline()
            s.cancel_event.set()

            try:
                clear_session(s.id)
            except Exception as e:
                log.error(f"[{s.id}] Error clearing session on reset: {e}")

            s.speech_buffer = []
            s.pre_buffer.clear()
            s.speech_active = False
            s.silence_time = 0.0
            s.speech_total_secs = 0.0
            s.mid_pause_secs = 0.0
            s.bc_sent = False
            s.spoken_so_far = ""
            s.gen_id = ""
            s.cancel_event = asyncio.Event()
            s.primary_speaker_embedding = None
            s.primary_speaker_enrolled = False
            s.enrollment_embeddings = []
            s.utterance_count = 0
            s.recent_primary_embeddings = []

            _reset_current_utterance_tracking(s)
            _reset_interrupt_tracking(s)
            s.set_state(SessionState.IDLE)

            await _safe_send_json(ws, {"type": "reset_ack"})
            log.info(f"[{s.id}] session reset")
            return None

        log.debug(f"[{s.id}] unknown control: {t!r}")
        return None

    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.error(f"[{s.id}] _handle_control error for type={t!r}: {e}")
        return None

