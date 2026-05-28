import asyncio
import logging
from typing import TypedDict
from app.config import WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_LANGUAGE

log = logging.getLogger("transcriber")


class Segment(TypedDict):
    speaker: str
    start:   float
    end:     float
    text:    str


def _merge_consecutive(segments: list[Segment]) -> list[Segment]:
    """Merge adjacent segments from the same speaker into utterances."""
    if not segments:
        return []

    merged: list[Segment] = []
    current = dict(segments[0])

    for seg in segments[1:]:
        if seg["speaker"] == current["speaker"]:
            current["text"] += " " + seg["text"].strip()
            current["end"] = seg["end"]
        else:
            merged.append(current)  # type: ignore[arg-type]
            current = dict(seg)

    merged.append(current)  # type: ignore[arg-type]
    return merged


def _transcribe_whisperx(audio_path: str) -> tuple[list[Segment], str]:
    import whisperx  # type: ignore

    log.info(f"[transcriber] Loading whisperX model ({WHISPER_MODEL_SIZE})")
    model = whisperx.load_model(WHISPER_MODEL_SIZE, WHISPER_DEVICE, language=WHISPER_LANGUAGE)
    audio = whisperx.load_audio(audio_path)

    result = model.transcribe(audio, batch_size=16)
    language = result.get("language", WHISPER_LANGUAGE)

    # Alignment
    align_model, metadata = whisperx.load_align_model(language_code=language, device=WHISPER_DEVICE)
    result = whisperx.align(result["segments"], align_model, metadata, audio, WHISPER_DEVICE)

    # Diarization (requires HuggingFace token for pyannote — skip if unavailable)
    try:
        import os
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=WHISPER_DEVICE)
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        else:
            log.warning("[transcriber] HF_TOKEN not set — skipping diarization, all segments labeled SPEAKER_00")
    except Exception as e:
        log.warning(f"[transcriber] Diarization failed ({e}) — proceeding without speaker labels")

    segments: list[Segment] = []
    for seg in result.get("segments", []):
        segments.append({
            "speaker": seg.get("speaker", "SPEAKER_00"),
            "start":   round(float(seg.get("start", 0)), 2),
            "end":     round(float(seg.get("end", 0)), 2),
            "text":    seg.get("text", "").strip(),
        })

    return _merge_consecutive(segments), language


def _transcribe_faster_whisper(audio_path: str) -> tuple[list[Segment], str]:
    from faster_whisper import WhisperModel  # type: ignore

    log.info(f"[transcriber] Fallback: faster-whisper ({WHISPER_MODEL_SIZE})")
    model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type="int8")
    raw_segs, info = model.transcribe(audio_path, language=WHISPER_LANGUAGE or None)

    segments: list[Segment] = []
    for i, seg in enumerate(raw_segs):
        segments.append({
            "speaker": "SPEAKER_00",
            "start":   round(seg.start, 2),
            "end":     round(seg.end, 2),
            "text":    seg.text.strip(),
        })

    return _merge_consecutive(segments), getattr(info, "language", WHISPER_LANGUAGE)


def _do_transcribe(audio_path: str) -> tuple[list[Segment], str]:
    try:
        return _transcribe_whisperx(audio_path)
    except ImportError:
        log.warning("[transcriber] whisperx not installed — falling back to faster-whisper")
        return _transcribe_faster_whisper(audio_path)


def segments_to_text(segments: list[Segment]) -> str:
    lines = []
    for seg in segments:
        ts = f"[{seg['start']:.1f}s – {seg['end']:.1f}s]"
        lines.append(f"{seg['speaker']} {ts}: {seg['text']}")
    return "\n".join(lines)


async def transcribe(audio_path: str) -> tuple[list[Segment], str]:
    """
    Transcribe audio_path with speaker diarization.
    Returns (segments, formatted_transcript_string).
    Runs in a thread to avoid blocking the event loop.
    """
    segments, language = await asyncio.to_thread(_do_transcribe, audio_path)
    transcript_text = segments_to_text(segments)
    log.info(f"[transcriber] Done — {len(segments)} utterances, language={language}")
    return segments, transcript_text
