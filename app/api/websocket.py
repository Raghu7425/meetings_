"""
Handles WebSocket connection for real-time audio communication.

- Accepts client connection and creates a session
- Sends session details (TTS engine and voice) to client
- Receives text (control) and audio messages from client

- Converts audio bytes to PCM and detects speech using VAD
- Handles user speaking and detects when speech ends
- Handles AI speaking and allows user interrupt (barge-in)

- Sends backchannel responses during user speech if needed
- Starts processing pipeline after user finishes speaking

- Keeps connection alive and cleans up session on disconnect

Manages the full flow of real-time user audio ↔ AI response.
"""


import time
import asyncio
import logging
import uuid
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.api.ws_pipeline import _launch_pipeline
from app.api.ws_audio import run_vad, _start_speech
from app.api.ws_output import _send_backchannel, _trigger_interrupt
from app.api.ws_session import SessionState, _reset_interrupt_tracking
from app.api.ws_speaker_flow import _check_current_speaker_if_needed, _check_interrupt_speaker
from app.api.ws_transport import _handle_control, _keepalive, _safe_send_json
from app.core.agent import clear_session, get_backchannel
from app.core.tts import get_edge_voice, get_engine_name
from app.config import (BC_MIN_SPEECH_BEFORE, BC_PAUSE_THRESHOLD, SAMPLE_RATE, SILENCE_LIMIT, 
                        SPEECH_THRESHOLD, BC_MIN_SPEECH_FALLBACK, BC_PAUSE_FALLBACK)


log = logging.getLogger("websocket")
router = APIRouter()


async def _send_session_start(ws: WebSocket, s: SessionState) -> bool:
    try:
        await ws.send_json(
            {
                "type": "session_start",
                "session_id": s.id,
                "tts_engine": get_engine_name(),
                "tts_voice": get_edge_voice(),
            }
        )
        return True

    except Exception as e:
        log.error(f"[{s.id}] Failed to send session_start: {e}")
        return False



async def _receive_message(ws: WebSocket, s: SessionState):
    try:
        return await ws.receive()

    except WebSocketDisconnect:
        raise

    except RuntimeError as e:
        msg_text = str(e)
        if "disconnect message has been received" in msg_text:
            raise WebSocketDisconnect()
        log.warning(f"[{s.id}] Receive issue: {e}")
        raise

    except Exception as e:
        log.warning(f"[{s.id}] Receive issue: {e}")
        raise



async def _handle_text_message(ws: WebSocket, s: SessionState, raw: str) -> None:
    try:
        action = await _handle_control(ws, s, raw)
        if action == "interrupt" and s.state == SessionState.SPEAKING:
            log.info(f"[{s.id}] client interrupt ignored; only verified primary speaker can barge in")

    except Exception as e:
        log.error(f"[{s.id}] _handle_control error: {e}")



def _decode_audio_chunk(s: SessionState, raw_bytes: bytes):

    if not raw_bytes:
        return None

    try:
        # # Convert raw int16 audio bytes to normalized float32 waveform (-1.0 to 1.0)
        return np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    except Exception as e:
        log.error(f"[{s.id}] Audio decode error: {e}")
        return None



async def _handle_speaking_state(ws: WebSocket, s: SessionState, pcm: np.ndarray, speech_prob: float, chunk_secs: float) -> None:

    if speech_prob > SPEECH_THRESHOLD:
        s.interrupt_buffer.append(pcm)
        s.interrupt_speech_secs += chunk_secs

        allow_interrupt = await _check_interrupt_speaker(ws, s)
        if allow_interrupt:
            log.info(
                f"[{s.id}] PRIMARY interrupt accepted "
                f"(score={s.current_speaker_score:.3f})"
            )
            await _trigger_interrupt(ws, s, "primary_barge_in")
            _start_speech(s, pcm)

    else:
        _reset_interrupt_tracking(s)



async def _maybe_send_backchannel(ws: WebSocket, s: SessionState) -> None:

    if not (
        s.primary_speaker_enrolled
        and s.current_speaker_checked
        and s.current_speaker_is_primary
        and not s.bc_sent
        and s.speech_total_secs >= max(BC_MIN_SPEECH_BEFORE, BC_MIN_SPEECH_FALLBACK)
        and s.mid_pause_secs >= max(BC_PAUSE_THRESHOLD, BC_PAUSE_FALLBACK)
        and s.silence_time < SILENCE_LIMIT
    ):
        return

    try:
        filler = get_backchannel(s.id)
        if filler:
            task = asyncio.ensure_future(_send_backchannel(ws, s, filler))
            task.add_done_callback(
                lambda t: log.warning(f"[{s.id}] BC error: {t.exception()}")
                if not t.cancelled() and t.exception()
                else None
            )
            s.bc_sent = True

    except Exception as e:
        log.error(f"[{s.id}] Backchannel error: {e}")



async def _handle_listening_state(ws: WebSocket, s: SessionState, pcm: np.ndarray, speech_prob: float, chunk_secs: float) -> None:

    if speech_prob > SPEECH_THRESHOLD:
        if not s.speech_active:
            try:
                pre = list(s.pre_buffer)
                s.pre_buffer.clear()
                _start_speech(s, pcm, pre_buffer=pre)
                await _safe_send_json(ws, {"type": "status", "text": "listening"})
            except Exception as e:
                log.error(f"[{s.id}] Error starting speech: {e}")
                return
        else:
            s.speech_buffer.append(pcm)
            s.silence_time = 0.0

        s.speech_total_secs += chunk_secs
        s.mid_pause_secs = 0.0

        should_continue = await _check_current_speaker_if_needed(ws, s)
        if not should_continue:
            return

    else:
        if not s.speech_active:
            s.pre_buffer.append(pcm)

        if s.speech_active:
            s.speech_buffer.append(pcm)
            s.silence_time += chunk_secs
            s.mid_pause_secs += chunk_secs

            await _maybe_send_backchannel(ws, s)

            if s.silence_time >= SILENCE_LIMIT:
                s.t_speech_end = time.monotonic()
                try:
                    _launch_pipeline(ws, s)
                except Exception as e:
                    log.error(f"[{s.id}] Pipeline launch error: {e}")



async def _cleanup_session(s: SessionState, keepalive_task: asyncio.Task) -> None:
    keepalive_task.cancel()
    s.cancel_pipeline()
    try:
        clear_session(s.id)
    except Exception as e:
        log.error(f"[{s.id}] Error clearing session: {e}")
    log.info(f"[{s.id}] cleaned up")



@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):

    await ws.accept()

    session_id = str(uuid.uuid4())[:8]
    s = SessionState(session_id)
    log.info(f"[{s.id}] connected")

    ok = await _send_session_start(ws, s)
    if not ok:
        return

    keepalive_task = asyncio.ensure_future(_keepalive(ws, s))

    try:
        while True:
            msg = await _receive_message(ws, s)

            if raw := msg.get("text"):
                await _handle_text_message(ws, s, raw)
                continue

            pcm = _decode_audio_chunk(s, msg.get("bytes"))
            if pcm is None:
                continue

            try:
                speech_prob = run_vad(pcm)
            except Exception as e:
                log.error(f"[{s.id}] VAD error: {e}")
                speech_prob = 0.0

            chunk_secs = len(pcm) / SAMPLE_RATE

            if s.state == SessionState.SPEAKING:
                await _handle_speaking_state(ws, s, pcm, speech_prob, chunk_secs)
                continue

            if s.state in (SessionState.PROCESSING, SessionState.INTERRUPTED):
                continue

            await _handle_listening_state(ws, s, pcm, speech_prob, chunk_secs)

    except WebSocketDisconnect:
        log.info(f"[{s.id}] disconnected")

    except Exception as e:
        log.exception(f"[{s.id}] receive loop error: {e}")

    finally:
        await _cleanup_session(s, keepalive_task)

