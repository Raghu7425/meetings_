"""
Handles speaker detection, enrollment, and verification.

- Collects speaker embeddings during initial phase (enrollment)
- Builds primary speaker profile after enough samples

- During user speech:
  • Checks if current speaker is the primary user
  • Ignores other speakers and stops processing

- During interrupt:
  • Verifies if interrupt is from primary speaker
  • Allows interrupt only for valid user

- Updates primary speaker embedding over time for better accuracy

This file ensures system responds only to the correct user voice.
"""


import logging
import numpy as np
from fastapi import WebSocket
from app.api.ws_transport import _safe_send_json
from app.api.ws_session import SessionState, _drop_current_utterance
from app.core.speaker import extract_embedding, classify_speaker, build_primary_embedding, average_embeddings
from app.config import (SAMPLE_RATE, BC_MIN_SPEECH_BEFORE, SPEAKER_THRESHOLD, 
                        MIN_SPEAKER_CHECK_SECS, MIN_INTERRUPT_CHECK_SECS, TARGET_SPEAKER_VERIFICATION)


log = logging.getLogger("websocket")


def _store_enrollment_embedding(s: SessionState, current_emb: np.ndarray | None) -> None:
    
    if current_emb is None or s.primary_speaker_enrolled:
        return

    if len(s.enrollment_embeddings) >= s.enrollment_target_count:
        return

    s.enrollment_embeddings.append(current_emb)
    collected = len(s.enrollment_embeddings)

    log.info(f"[{s.id}] enrollment sample stored ({collected}/{s.enrollment_target_count})")

    if collected == s.enrollment_target_count:
        primary_emb = build_primary_embedding(s.enrollment_embeddings)

        if primary_emb is not None:
            s.primary_speaker_embedding = primary_emb
            s.primary_speaker_enrolled = True

            # seed rolling window with enrollment embeddings
            s.recent_primary_embeddings = list(s.enrollment_embeddings)
            log.info(f"[{s.id}] PRIMARY speaker enrolled using best enrollment samples")

        else:
            log.warning(f"[{s.id}] failed to build primary embedding from enrollment samples")



async def _check_current_speaker_if_needed(ws: WebSocket, s: SessionState) -> bool:

    if s.current_speaker_checked:
        return s.current_speaker_is_primary

    if not s.primary_speaker_enrolled:
        return True

    if s.speech_total_secs < max(BC_MIN_SPEECH_BEFORE, MIN_SPEAKER_CHECK_SECS):
        return True

    if len(s.speech_buffer) < 2:
        return True

    try:
        audio = np.concatenate(s.speech_buffer)
        current_emb = await __import__("asyncio").to_thread(extract_embedding, audio, SAMPLE_RATE)
        if current_emb is None:
            return True

        is_primary, score = classify_speaker(
            current_emb,
            s.primary_speaker_embedding,
            threshold=SPEAKER_THRESHOLD,
        )

        s.current_speaker_checked = True
        s.current_speaker_is_primary = is_primary
        s.current_speaker_score = score

        if not is_primary:
            log.info(f"[{s.id}] SECONDARY speaker skipped during listening (score={score:.3f})")
            await _safe_send_json(ws, {"type": "speaker", "label": "secondary"})
            _drop_current_utterance(s)
            await _safe_send_json(ws, {"type": "status", "text": "idle"})
            return False

        return True

    except Exception as e:
        log.debug(f"[{s.id}] current speaker check failed: {e}")
        return True



async def _check_interrupt_speaker(ws: WebSocket, s: SessionState) -> bool:

    if not s.primary_speaker_enrolled:
        return False

    if s.interrupt_check_done:
        return s.current_speaker_is_primary

    if s.interrupt_speech_secs < MIN_INTERRUPT_CHECK_SECS:
        return False

    if len(s.interrupt_buffer) < 2:
        return False

    try:
        audio = np.concatenate(s.interrupt_buffer)
        current_emb = await __import__("asyncio").to_thread(extract_embedding, audio, SAMPLE_RATE)
        if current_emb is None:
            return False

        is_primary, score = classify_speaker(
            current_emb,
            s.primary_speaker_embedding,
            threshold=SPEAKER_THRESHOLD,
        )

        s.interrupt_check_done = True
        s.current_speaker_checked = True
        s.current_speaker_is_primary = is_primary
        s.current_speaker_score = score

        if not is_primary:
            log.info(f"[{s.id}] secondary speaker ignored during speaking (score={score:.3f})")

        return is_primary

    except Exception as e:
        log.debug(f"[{s.id}] interrupt speaker check failed: {e}")
        return False



def _update_primary_embedding(s: SessionState, current_emb, score: float) -> None:

    if current_emb is None:
        return

    if not s.primary_speaker_enrolled:
        return

    if score < s.primary_update_min_score:
        return

    try:
        if TARGET_SPEAKER_VERIFICATION:
            s.recent_primary_embeddings.append(current_emb)
            if len(s.recent_primary_embeddings) > s.primary_update_window:
                s.recent_primary_embeddings.pop(0)

        updated = average_embeddings(s.recent_primary_embeddings)
        if updated is not None:
            s.primary_speaker_embedding = updated
            log.info(
                f"[{s.id}] primary embedding updated "
                f"(score={score:.3f}, window={len(s.recent_primary_embeddings)})"
            )

    except Exception as e:
        log.error(f"[{s.id}] rolling primary update failed: {e}")

