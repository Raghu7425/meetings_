"""
Handles AI response output and audio streaming.

- Decides when to split and send text for TTS (flush logic)
- Converts AI text response into audio using TTS
- Sends small filler responses (backchannel) during user speech
- Streams AI response text and audio chunk by chunk

- Handles interrupt:
  • Stops current response immediately
  • Clears buffers and resets state

- Updates session data after response is completed
- Manages full AI speaking flow from text generation → audio output

This file controls how AI responses are generated, streamed, and stopped.
"""


import re
import time
import asyncio
import logging
from fastapi import WebSocket
from app.api.ws_session import SessionState, _reset_current_utterance_tracking, _reset_interrupt_tracking
from app.api.ws_transport import _safe_send_json, _safe_send_bytes
from app.core.agent import chat_stream, get_session_state, truncate_last_assistant
from app.core.tts import synthesize_chunks
from app.config import TTS_FLUSH_CHARS, SESSION_UPDATE_DELAY


log = logging.getLogger("websocket")
_SENTENCE_END = re.compile(r'[.!?;:]["\']?$')


def _should_flush(buf: str) -> bool:
    try:
        s = buf.strip()
        if not s:
            return False

        if _SENTENCE_END.search(s):
            return True

        if len(s) >= TTS_FLUSH_CHARS:
            if s[-1] in " \t.!?;:,":
                return True
            if " " in s[-20:]:
                return True

        return False

    except Exception as e:
        log.error(f"[TTS flush] _should_flush error: {e}")
        return False



def _split_at_boundary(buf: str) -> tuple[str, str]:
    try:
        s = buf.strip()
        if not s:
            return "", buf

        matches = list(_SENTENCE_END.finditer(s))
        if matches:
            split_pos = matches[-1].end()
            return s[:split_pos].strip(), s[split_pos:]

        if len(s) >= TTS_FLUSH_CHARS:
            last_space = s.rfind(" ")
            if last_space != -1:
                return s[:last_space].strip(), s[last_space + 1:]

        return "", buf

    except Exception as e:
        log.error(f"[TTS flush] _split_at_boundary error: {e}")
        return "", buf



async def _send_backchannel(ws: WebSocket, s: SessionState, filler: str):

    if s.state != SessionState.LISTENING:
        return

    try:
        audio_bytes = await synthesize_chunks(filler)
        if audio_bytes and s.state == SessionState.LISTENING:
            await _safe_send_json(ws, {"type": "backchannel", "text": filler})
            await _safe_send_bytes(ws, audio_bytes)
            log.info(f"[{s.id}] backchannel: '{filler}'")

    except asyncio.CancelledError:
        pass
    except Exception as e:
        log.debug(f"[{s.id}] BC error: {e}")



async def _send_tts_chunk(ws: WebSocket, s: SessionState, cancel: asyncio.Event, text: str, gen_id: str):

    if cancel.is_set():
        return

    try:
        session_voice = get_session_state(s.id).get("voice") or None
    except Exception:
        session_voice = None

    try:
        audio_bytes = await synthesize_chunks(text, cancel=cancel, voice=session_voice)
        if not audio_bytes or cancel.is_set():
            return

        await _safe_send_bytes(ws, audio_bytes)
        s.spoken_so_far += text + " "

    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.error(f"[{s.id}] gen={gen_id} TTS error: {e}")



async def _deferred_session_update(ws: WebSocket, s: SessionState, delay: float = SESSION_UPDATE_DELAY):
    try:
        await asyncio.sleep(delay)
        await _safe_send_json(ws, {"type": "session_update", "session": get_session_state(s.id)})

    except asyncio.CancelledError:
        pass
    except Exception as e:
        log.debug(f"[{s.id}] _deferred_session_update error: {e}")



async def _trigger_interrupt(ws: WebSocket, s: SessionState, reason: str = "unknown"):
    try:
        log.warning(f"[{s.id}] INTERRUPT reason={reason} gen={s.gen_id}")

        already_cancelled = s.cancel_event.is_set()
        s.cancel_event.set()
        s.cancel_pipeline()

        if not already_cancelled and s.spoken_so_far.strip():
            truncate_last_assistant(s.id, s.spoken_so_far)

        await _safe_send_json(
            ws,
            {
                "type": "interrupt",
                "reason": reason,
                "gen_id": s.gen_id,
            },
        )

        s.speech_active = False
        s.speech_buffer = []
        s.silence_time = 0.0
        s.speech_total_secs = 0.0
        s.mid_pause_secs = 0.0
        s.bc_sent = False
        s.spoken_so_far = ""
        _reset_current_utterance_tracking(s)
        _reset_interrupt_tracking(s)
        s.set_state(SessionState.LISTENING)

    except Exception as e:
        log.error(f"[{s.id}] _trigger_interrupt error: {e}")
        try:
            s.set_state(SessionState.LISTENING)
        except Exception:
            pass



async def _run_speaking(ws: WebSocket, s: SessionState, user_text: str, gen_id: str, stt_ms: int = 0):

    cancel = s.cancel_event
    s.set_state(SessionState.SPEAKING)
    await _safe_send_json(ws, {"type": "status", "text": "speaking"})

    tts_buf = ""
    llm_ms = 0
    tts_ms = 0
    first_token = True
    first_audio = True
    t_llm_start = time.monotonic()

    try:
        async for token in chat_stream(user_text, s.id):
            if cancel.is_set():
                return

            if first_token:
                llm_ms = int((time.monotonic() - t_llm_start) * 1000)
                first_token = False

            if not isinstance(token, str):
                continue

            tts_buf += token
            await _safe_send_json(ws, {"type": "ai_stream", "text": token})

            if _should_flush(tts_buf):
                chunk, tts_buf = _split_at_boundary(tts_buf)
                if chunk and not cancel.is_set():
                    t_tts = time.monotonic()
                    await _send_tts_chunk(ws, s, cancel, chunk, gen_id)
                    if first_audio and s.spoken_so_far:
                        tts_ms = int((time.monotonic() - t_tts) * 1000)
                        first_audio = False

        if tts_buf.strip() and not cancel.is_set():
            t_tts = time.monotonic()
            await _send_tts_chunk(ws, s, cancel, tts_buf.strip(), gen_id)
            if first_audio and s.spoken_so_far:
                tts_ms = int((time.monotonic() - t_tts) * 1000)

    except asyncio.CancelledError:
        return
    
    except Exception as e:
        log.error(f"[{s.id}] gen={gen_id} error: {e}")
        s.set_state(SessionState.IDLE)
        await _safe_send_json(ws, {"type": "status", "text": "idle"})
        await _safe_send_json(ws, {"type": "error", "text": "Something went wrong. Please speak again."})
        return

    if not cancel.is_set():
        log.info(f"[{s.id}] gen={gen_id} done. STT={stt_ms}ms LLM={llm_ms}ms TTS={tts_ms}ms")
        s.set_state(SessionState.IDLE)
        await _safe_send_json(ws, {"type": "status", "text": "idle"})
        await _safe_send_json(
            ws,
            {
                "type": "ai_done",
                "session": get_session_state(s.id),
                "latency": {
                    "stt_ms": stt_ms,
                    "llm_ms": llm_ms,
                    "tts_ms": tts_ms,
                    "total_ms": stt_ms + llm_ms + tts_ms,
                },
            },
        )
        asyncio.ensure_future(_deferred_session_update(ws, s))

