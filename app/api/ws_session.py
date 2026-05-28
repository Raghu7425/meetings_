"""
Defines session state and all variables used during a WebSocket connection.

- Stores session id and current state (idle, listening, processing, speaking)
- Maintains audio buffers and tracks speech activity and silence
- Tracks timing values for speech, STT, and response generation
- Manages current AI response (generation id, cancel event, pipeline task)

- Handles speaker enrollment:
  • Collects embeddings to identify primary speaker

- Handles speaker verification:
  • Checks if current speaker is the same user

- Tracks interrupt data when user tries to stop AI speech
- Provides helper functions to reset tracking and clear current data

This file stores and manages all runtime data for a single session.
"""


import asyncio
import logging
import uuid
import numpy as np
from collections import deque
from app.config import PRIMARY_ENROLLMENT_COUNT, PRIMARY_UPDATE_WINDOW, PRIMARY_UPDATE_MIN_SCORE


log = logging.getLogger("websocket")


class SessionState:
    
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"

    def __init__(self, session_id: str):
        self.id = session_id
        self.state = self.IDLE

        self.speech_buffer: list[np.ndarray] = []
        self.speech_active = False
        self.pre_buffer: deque[np.ndarray] = deque(maxlen=8)
        self.silence_time = 0.0

        self.speech_total_secs = 0.0
        self.mid_pause_secs = 0.0
        self.bc_sent = False

        self.cancel_event: asyncio.Event = asyncio.Event()
        self.gen_id: str = ""
        self.spoken_so_far: str = ""
        self.pipeline_task: asyncio.Task | None = None

        self.pong_received: asyncio.Event = asyncio.Event()
        self.pong_received.set()

        self.t_speech_end = 0.0
        self.t_stt_done = 0.0
        self.t_first_token = 0.0
        self.t_first_audio = 0.0

        # speaker enrollment / verification
        self.primary_speaker_embedding = None
        self.primary_speaker_enrolled = False
        self.enrollment_embeddings: list[np.ndarray] = []
        self.enrollment_target_count = PRIMARY_ENROLLMENT_COUNT
        self.utterance_count = 0

        # rolling primary update
        self.recent_primary_embeddings: list[np.ndarray] = []
        self.primary_update_window = PRIMARY_UPDATE_WINDOW
        self.primary_update_min_score = PRIMARY_UPDATE_MIN_SCORE

        # current utterance early-check state
        self.current_speaker_checked = False
        self.current_speaker_is_primary = True
        self.current_speaker_score = 0.0

        # interrupt state while AI is speaking
        self.interrupt_buffer: list[np.ndarray] = []
        self.interrupt_speech_secs = 0.0
        self.interrupt_check_done = False

    def new_generation(self) -> str:
        try:
            self.cancel_event = asyncio.Event()
            self.gen_id = str(uuid.uuid4())[:8]
            self.spoken_so_far = ""
            self.t_speech_end = 0.0
            self.t_stt_done = 0.0
            self.t_first_token = 0.0
            self.t_first_audio = 0.0
            return self.gen_id
        except Exception as e:
            log.error(f"[{self.id}] new_generation error: {e}")
            self.gen_id = "err"
            return self.gen_id

    def set_state(self, ns: str):
        try:
            if ns != self.state:
                log.info(f"[{self.id}] {self.state} → {ns}")
                self.state = ns
        except Exception as e:
            log.error(f"[{self.id}] set_state error: {e}")

    def cancel_pipeline(self):
        try:
            if self.pipeline_task and not self.pipeline_task.done():
                self.pipeline_task.cancel()
            self.pipeline_task = None
        except Exception as e:
            log.error(f"[{self.id}] cancel_pipeline error: {e}")


def _reset_current_utterance_tracking(s: SessionState):
    s.current_speaker_checked = False
    s.current_speaker_is_primary = True
    s.current_speaker_score = 0.0


def _reset_interrupt_tracking(s: SessionState):
    s.interrupt_buffer = []
    s.interrupt_speech_secs = 0.0
    s.interrupt_check_done = False


def _drop_current_utterance(s: SessionState):
    s.speech_active = False
    s.speech_buffer = []
    s.silence_time = 0.0
    s.speech_total_secs = 0.0
    s.mid_pause_secs = 0.0
    s.bc_sent = False
    _reset_current_utterance_tracking(s)
    s.set_state(SessionState.IDLE)

