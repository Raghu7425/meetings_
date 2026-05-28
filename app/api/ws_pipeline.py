"""
Handles full processing pipeline after user finishes speaking.

- Takes collected audio and prepares it for processing
- Converts audio to text using STT (speech-to-text)
- Extracts speaker embedding from audio

- During initial phase:
  • Collects embeddings to identify primary speaker (enrollment)
  
- After enrollment:
  • Verifies if current speaker is the primary user
  • Ignores other speakers

- Validates transcript before continuing
- Sends transcript to client
- Starts AI response generation and speaking

This file controls the flow: audio → speaker check → text → AI response.
"""


import time
import asyncio
import logging
import numpy as np
from fastapi import WebSocket
from app.api.ws_session import SessionState
from app.api.ws_transport import _safe_send_json
from app.api.ws_speaker_flow import _store_enrollment_embedding, _update_primary_embedding
from app.api.ws_output import _run_speaking
from app.core.speaker import extract_embedding, classify_speaker
from app.core.stt import is_usable_transcript, transcribe_np
from app.config import MAX_AUDIO_SECS, MIN_SPEECH_SECS, SAMPLE_RATE, SPEAKER_THRESHOLD


log = logging.getLogger("websocket")


def _launch_pipeline(ws: WebSocket, s: SessionState):
    try:
        s.cancel_pipeline()
        task = asyncio.ensure_future(_process_utterance(ws, s))
        s.pipeline_task = task

        def _on_done(t: asyncio.Task):
            s.pipeline_task = None
            if not t.cancelled() and t.exception():
                log.error(f"[{s.id}] pipeline exception: {t.exception()}")
                s.set_state(SessionState.IDLE)
                asyncio.ensure_future(_safe_send_json(ws, {"type": "status", "text": "idle"}))

        task.add_done_callback(_on_done)

    except Exception as e:
        log.error(f"[{s.id}] _launch_pipeline error: {e}")



async def _finalize_utterance_audio(ws: WebSocket, s: SessionState) -> tuple[np.ndarray | None, float]:

    s.speech_active = False

    if not s.speech_buffer:
        log.info(f"[{s.id}] no speech_buffer -> idle")
        s.set_state(SessionState.IDLE)
        await _safe_send_json(ws, {"type": "status", "text": "idle"})
        return None, 0.0

    try:
        audio = np.concatenate(s.speech_buffer)
    except ValueError as e:
        log.error(f"[{s.id}] Failed to concatenate speech buffers: {e}")
        s.set_state(SessionState.IDLE)
        await _safe_send_json(ws, {"type": "status", "text": "idle"})
        return None, 0.0
    finally:
        s.speech_buffer = []

    dur = len(audio) / SAMPLE_RATE
    log.info(f"[{s.id}] utterance duration={dur:.3f}s")

    if dur < MIN_SPEECH_SECS:
        log.info(f"[{s.id}] utterance too short -> idle")
        s.set_state(SessionState.IDLE)
        await _safe_send_json(ws, {"type": "status", "text": "idle"})
        return None, 0.0

    if dur > MAX_AUDIO_SECS:
        audio = audio[-int(SAMPLE_RATE * MAX_AUDIO_SECS):]
        log.info(f"[{s.id}] audio trimmed to last {MAX_AUDIO_SECS}s for STT")

    return audio, dur



async def _extract_current_embedding(ws: WebSocket, s: SessionState, audio: np.ndarray):
    try:
        current_emb = await asyncio.to_thread(extract_embedding, audio, SAMPLE_RATE)

        if current_emb is None:
            log.warning(f"[{s.id}] speaker embedding failed")
            s.set_state(SessionState.IDLE)
            await _safe_send_json(ws, {"type": "status", "text": "idle"})
            return None

        return current_emb

    except Exception as e:
        log.error(f"[{s.id}] speaker embedding error: {e}")
        s.set_state(SessionState.IDLE)
        await _safe_send_json(ws, {"type": "status", "text": "idle"})
        return None



async def _run_stt_and_validate(ws: WebSocket, s: SessionState, audio: np.ndarray, *, phase: str) -> tuple[str | None, int]:

    s.set_state(SessionState.PROCESSING)
    await _safe_send_json(ws, {"type": "status", "text": "thinking"})

    t_stt_start = time.monotonic()
    try:
        text = await asyncio.to_thread(transcribe_np, audio)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.error(f"[{s.id}] STT error during {phase}: {e}")
        s.set_state(SessionState.IDLE)
        await _safe_send_json(ws, {"type": "status", "text": "idle"})
        return None, 0

    s.t_stt_done = time.monotonic()
    stt_ms = int((s.t_stt_done - t_stt_start) * 1000)

    if not text:
        log.info(f"[{s.id}] STT empty text during {phase} -> idle")
        s.set_state(SessionState.IDLE)
        await _safe_send_json(ws, {"type": "status", "text": "idle"})
        return None, 0

    if not is_usable_transcript(text):
        log.info(f"[{s.id}] STT unusable text during {phase} -> {text!r}")
        s.set_state(SessionState.IDLE)
        await _safe_send_json(ws, {"type": "status", "text": "idle"})
        return None, 0

    return text, stt_ms



async def _handle_enrollment_phase(ws: WebSocket, s: SessionState, current_emb, text: str, stt_ms: int) -> bool:

    s.utterance_count += 1
    log.info(f"[{s.id}] valid enrollment utterance_count={s.utterance_count}")

    _store_enrollment_embedding(s, current_emb)

    collected = len(s.enrollment_embeddings)
    if s.primary_speaker_enrolled:
        await _safe_send_json(
            ws,
            {
                "type": "speaker",
                "label": "primary_enrolled",
                "accepted": collected,
                "target": s.enrollment_target_count,
            },
        )
        log.info(f"[{s.id}] primary enrolled, collected={collected}")

    else:
        await _safe_send_json(
            ws,
            {
                "type": "speaker",
                "label": "enrollment",
                "accepted": collected,
                "target": s.enrollment_target_count,
            },
        )
        log.info(f"[{s.id}] enrollment progress {collected}/{s.enrollment_target_count}")

    log.info(f"[{s.id}] user speech detected ({stt_ms}ms STT, len={len(text)})")
    await _safe_send_json(ws, {"type": "transcript", "text": text, "stt_ms": stt_ms})
    return True



async def _handle_verification_phase(ws: WebSocket, s: SessionState, current_emb) -> bool:
    try:
        is_primary, score = classify_speaker(
            current_emb,
            s.primary_speaker_embedding,
            threshold=SPEAKER_THRESHOLD,
        )

        if not is_primary:
            log.info(f"[{s.id}] SECONDARY speaker ignored (score={score:.3f})")
            s.set_state(SessionState.IDLE)
            await _safe_send_json(ws, {"type": "speaker", "label": "secondary"})
            await _safe_send_json(ws, {"type": "status", "text": "idle"})
            return False

        log.info(f"[{s.id}] PRIMARY speaker accepted (score={score:.3f})")
        _update_primary_embedding(s, current_emb, score)

        await _safe_send_json(
            ws,
            {
                "type": "speaker",
                "label": "primary",
                "score": round(score, 3),
            },
        )
        return True

    except Exception as e:
        log.error(f"[{s.id}] speaker verification error: {e}")
        s.set_state(SessionState.IDLE)
        await _safe_send_json(ws, {"type": "status", "text": "idle"})
        return False



async def _process_utterance(ws: WebSocket, s: SessionState):
    try:
        audio, _dur = await _finalize_utterance_audio(ws, s)
        if audio is None:
            return

        current_emb = await _extract_current_embedding(ws, s, audio)
        if current_emb is None:
            return

        # Enrollment phase
        if not s.primary_speaker_enrolled:
            text, stt_ms = await _run_stt_and_validate(ws, s, audio, phase="enrollment")
            if not text:
                return

            ok = await _handle_enrollment_phase(ws, s, current_emb, text, stt_ms)
            if not ok:
                return

            gen_id = s.new_generation()
            await _run_speaking(ws, s, text, gen_id, stt_ms)
            return

        # Verification phase
        ok = await _handle_verification_phase(ws, s, current_emb)
        if not ok:
            return

        text, stt_ms = await _run_stt_and_validate(ws, s, audio, phase="verification")
        if not text:
            return

        log.info(f"[{s.id}] user speech detected ({stt_ms}ms STT, len={len(text)})")
        await _safe_send_json(ws, {"type": "transcript", "text": text, "stt_ms": stt_ms})

        gen_id = s.new_generation()
        await _run_speaking(ws, s, text, gen_id, stt_ms)

    except asyncio.CancelledError:
        raise

    except Exception as e:
        log.error(f"[{s.id}] _process_utterance error: {e}")
        s.set_state(SessionState.IDLE)
        await _safe_send_json(ws, {"type": "status", "text": "idle"})

