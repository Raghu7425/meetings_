"""
Speech-to-text using faster-whisper (CTranslate2-optimised Whisper).

Public interface is unchanged:
  get_model()                          → loads/returns singleton WhisperModel
  transcribe_np(audio_np, sample_rate) → str
  is_usable_transcript(text)           → bool
"""

import logging
import numpy as np
from app.core.audio_preprocessing import denoise_chunk, is_noise_only
from app.config import (
    SAMPLE_RATE, STT_MODEL_NAME, STT_DEVICE, STT_COMPUTE_TYPE,
    STT_LANGUAGE, STT_REJECT_NON_ENGLISH, STT_BEAM_SIZE,
)

log = logging.getLogger("stt")

_model = None

# Unicode script ranges that indicate non-Latin scripts
_NON_LATIN_SCRIPTS: tuple[tuple[int, int], ...] = (
    (0x0900, 0x097F),  # Devanagari
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0B00, 0x0B7F),  # Oriya
    (0x0C00, 0x0C7F),  # Telugu
    (0x0D00, 0x0D7F),  # Malayalam
    (0x0E00, 0x0E7F),  # Thai
    (0x0F00, 0x0FFF),  # Tibetan
    (0x1000, 0x109F),  # Myanmar
    (0x3000, 0x9FFF),  # CJK, Hiragana, Katakana
    (0xAC00, 0xD7AF),  # Hangul
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x0600, 0x06FF),  # Arabic
    (0x0590, 0x05FF),  # Hebrew
    (0x10A0, 0x10FF),  # Georgian
    (0x0400, 0x04FF),  # Cyrillic
    (0x0370, 0x03FF),  # Greek
)

_NOISE_TOKENS: frozenset[str] = frozenset({
    ".", "..", "...", "!", "?", ",", "-", "--",
    "you", "the", "a", "i", "oh", "ah", "uh", "um", "hmm", "hm", "mm",
    "ok", "okay", "yeah", "yes", "no", "hi", "hey", "bye", "bye.",
    "thanks for watching", "thanks for watching.",
    "please subscribe", "don't forget to subscribe",
    "like and subscribe", "hit the bell",
    "see you in the next video", "see you next time",
    "[music]", "[ music ]", "[applause]", "[ applause ]",
    "(music)", "(applause)", "[silence]", "[ silence ]",
    "[laughter]", "[ laughter ]", "(laughter)",
    "[background noise]", "[noise]", "[inaudible]",
    "subtitles by", "captions by", "transcribed by",
    "♪", "♫",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _contains_non_latin(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        for lo, hi in _NON_LATIN_SCRIPTS:
            if lo <= cp <= hi:
                return True
    return False


def _enforce_english(text: str) -> str:
    if not text:
        return ""
    if _contains_non_latin(text):
        log.warning("[STT] Rejected non-Latin transcript: %r", text[:80])
        return ""
    alpha = [c for c in text if c.isalpha()]
    if alpha and (sum(1 for c in alpha if ord(c) < 128) / len(alpha)) < 0.5:
        log.warning("[STT] Low ASCII-alpha ratio — likely non-English, rejecting: %r", text[:80])
        return ""
    return text


def _resolve_device_and_compute() -> tuple[str, str]:
    """Return (device, compute_type) for faster-whisper."""
    import torch

    hint = (STT_DEVICE or "auto").strip().lower()
    if hint == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = hint

    ct_hint = (STT_COMPUTE_TYPE or "auto").strip().lower()
    if ct_hint != "auto":
        compute_type = ct_hint
    else:
        compute_type = "float16" if device == "cuda" else "int8"

    return device, compute_type


# ── Model singleton ───────────────────────────────────────────────────────────

def get_model():
    global _model

    if _model is not None:
        return _model

    from faster_whisper import WhisperModel  # type: ignore

    device, compute_type = _resolve_device_and_compute()
    log.info("[STT] Loading faster-whisper model=%s device=%s compute=%s",
             STT_MODEL_NAME, device, compute_type)

    _model = WhisperModel(STT_MODEL_NAME, device=device, compute_type=compute_type)

    log.info("[STT] faster-whisper ready  language=%r reject_non_english=%s",
             STT_LANGUAGE, STT_REJECT_NON_ENGLISH)
    return _model


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe_np(audio_np: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """
    Transcribe a numpy float32 PCM array.
    Returns the transcript string, or "" if audio is silence/noise/invalid.
    """
    if audio_np is None or not isinstance(audio_np, np.ndarray):
        log.warning("[STT] Invalid input type: %s", type(audio_np))
        return ""

    if audio_np.size == 0:
        return ""

    if audio_np.ndim == 2:
        audio_np = audio_np.mean(axis=1)

    if audio_np.ndim != 1:
        log.warning("[STT] Unexpected audio shape %s", getattr(audio_np, "shape", None))
        return ""

    if audio_np.dtype != np.float32:
        try:
            audio_np = audio_np.astype(np.float32)
        except Exception as e:
            log.error("[STT] dtype cast failed: %s", e)
            return ""

    if not np.isfinite(audio_np).all():
        log.warning("[STT] NaN/Inf in audio — skipping")
        return ""

    peak = float(np.abs(audio_np).max())
    if peak > 1.0:
        audio_np = audio_np / peak

    try:
        audio_np = denoise_chunk(audio_np, sample_rate)
        if is_noise_only(audio_np, sample_rate):
            return ""
    except Exception as e:
        log.error("[STT] Preprocessing error: %s", e)
        return ""

    try:
        model = get_model()
    except Exception as e:
        log.error("[STT] Model unavailable: %s", e)
        return ""

    try:
        # faster-whisper accepts float32 numpy arrays at 16 kHz directly
        segments_gen, info = model.transcribe(
            audio_np,
            language=STT_LANGUAGE or None,
            beam_size=STT_BEAM_SIZE,
            vad_filter=True,        # skip silent regions
            word_timestamps=False,
        )

        parts = [seg.text.strip() for seg in segments_gen if seg.text.strip()]
        transcript = " ".join(parts)

        if transcript:
            duration_s = len(audio_np) / max(sample_rate, 1)
            log.debug("[STT] (%.1fs | lang=%s) → %r",
                      duration_s, getattr(info, "language", "?"), transcript)

        if STT_REJECT_NON_ENGLISH:
            transcript = _enforce_english(transcript)

        return transcript

    except MemoryError:
        log.error("[STT] OOM during transcription — audio too long?")
        return ""
    except Exception as e:
        log.error("[STT] Transcription error: %s", e)
        return ""


# ── Validation ────────────────────────────────────────────────────────────────

def is_usable_transcript(text: str) -> bool:
    try:
        if not isinstance(text, str):
            return False
        stripped = text.strip()
        if len(stripped) < 2:
            return False
        cleaned = stripped.lower().rstrip(".!?,;:")
        if cleaned in _NOISE_TOKENS:
            log.debug("[STT] Discarded hallucination: %r", text)
            return False
        if sum(c.isalpha() for c in stripped) < 3:
            return False
        if _contains_non_latin(stripped):
            return False
        return True
    except Exception as e:
        log.error("[STT] is_usable_transcript error: %s", e)
        return False
