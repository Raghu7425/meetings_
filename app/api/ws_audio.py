"""
Handles audio processing and speech detection.

- Loads Silero VAD model during startup
- Runs VAD on audio chunks to detect speech probability
- Splits audio into frames and finds peak speech score

- Starts a new speech session when user begins speaking
- Resets buffers, silence timers, and tracking values

Used to detect when user starts speaking and manage speech data.
"""


import logging
import torch
import numpy as np
from silero_vad import load_silero_vad
from app.api.ws_session import SessionState, _reset_current_utterance_tracking, _reset_interrupt_tracking
from app.config import SAMPLE_RATE, VAD_FRAME_SIZE


log = logging.getLogger("websocket")


try:
    log.info("Loading Silero VAD…")
    vad_model = load_silero_vad()
    log.info("VAD ready.")
except Exception as e:
    log.error(f"[startup] Failed to load Silero VAD: {e}")
    raise


def run_vad(pcm: np.ndarray) -> float:
    try:
        if pcm is None or len(pcm) == 0:
            return 0.0

        frame_size = VAD_FRAME_SIZE
        peak = 0.0

        with torch.no_grad():
            for i in range(0, len(pcm) - frame_size + 1, frame_size):
                try:
                    prob = vad_model(
                        torch.from_numpy(pcm[i:i + frame_size]).unsqueeze(0),
                        SAMPLE_RATE,
                    ).item()
                    peak = max(peak, prob)
                except RuntimeError as e:
                    log.warning(f"[VAD] Frame error at offset {i}: {e}")
                    continue

        return peak

    except Exception as e:
        log.error(f"[VAD] Unexpected error: {e}")
        return 0.0



def _start_speech(s: SessionState, first_chunk: np.ndarray, pre_buffer: list | None = None):
    try:
        s.speech_active = True
        s.speech_buffer = (pre_buffer or []) + [first_chunk]
        s.silence_time = 0.0
        s.speech_total_secs = 0.0
        s.mid_pause_secs = 0.0
        s.bc_sent = False
        _reset_current_utterance_tracking(s)
        _reset_interrupt_tracking(s)
        s.set_state(SessionState.LISTENING)

    except Exception as e:
        log.error(f"[{s.id}] _start_speech error: {e}")

