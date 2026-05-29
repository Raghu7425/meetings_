import asyncio
import logging
import time
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
    import os
    import whisperx  # type: ignore

    job_start = time.perf_counter()

    # 1. Load model
    log.info("transcriber step=loading_model model=%s device=%s", WHISPER_MODEL_SIZE, WHISPER_DEVICE)
    t0 = time.perf_counter()
    model = whisperx.load_model(WHISPER_MODEL_SIZE, WHISPER_DEVICE, language=WHISPER_LANGUAGE)
    log.info("transcriber step=model_ready elapsed=%.1fs", time.perf_counter() - t0)

    # 2. Load audio
    log.info("transcriber step=loading_audio path=%s", audio_path)
    t0 = time.perf_counter()
    audio = whisperx.load_audio(audio_path)
    audio_dur = len(audio) / 16_000
    log.info("transcriber step=audio_ready duration_min=%.1f elapsed=%.1fs",
             audio_dur / 60, time.perf_counter() - t0)

    # 3. ASR transcription
    log.info("transcriber step=transcribing batch_size=16")
    t0 = time.perf_counter()
    result = model.transcribe(audio, batch_size=16)
    language = result.get("language", WHISPER_LANGUAGE)
    n_segs = len(result.get("segments", []))
    log.info("transcriber step=asr_done segments=%d lang=%s elapsed=%.1fs",
             n_segs, language, time.perf_counter() - t0)

    # 4. Word-level alignment
    log.info("transcriber step=aligning lang=%s", language)
    t0 = time.perf_counter()
    align_model, metadata = whisperx.load_align_model(
        language_code=language, device=WHISPER_DEVICE
    )
    result = whisperx.align(result["segments"], align_model, metadata, audio, WHISPER_DEVICE)
    log.info("transcriber step=alignment_done elapsed=%.1fs", time.perf_counter() - t0)

    # 5. Speaker diarization
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            log.info("transcriber step=loading_diarizer device=%s", WHISPER_DEVICE)
            t0 = time.perf_counter()
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token, device=WHISPER_DEVICE
            )
            log.info("transcriber step=diarizer_loaded elapsed=%.1fs", time.perf_counter() - t0)

            log.info("transcriber step=diarizing audio_min=%.1f", audio_dur / 60)
            t0 = time.perf_counter()
            diarize_segments = diarize_model(audio)
            log.info("transcriber step=diarization_done elapsed=%.1fs", time.perf_counter() - t0)

            log.info("transcriber step=assigning_speakers")
            t0 = time.perf_counter()
            result = whisperx.assign_word_speakers(diarize_segments, result)
            log.info("transcriber step=speakers_assigned elapsed=%.1fs", time.perf_counter() - t0)
        except Exception as exc:
            log.warning("transcriber diarization_failed error=%s — no speaker labels", exc)
    else:
        log.warning("transcriber HF_TOKEN not set — skipping diarization, all SPEAKER_00")

    # 6. Build segment list
    segments: list[Segment] = []
    for seg in result.get("segments", []):
        segments.append({
            "speaker": seg.get("speaker", "SPEAKER_00"),
            "start":   round(float(seg.get("start", 0)), 2),
            "end":     round(float(seg.get("end", 0)), 2),
            "text":    seg.get("text", "").strip(),
        })

    log.info(
        "transcriber complete engine=whisperx segments=%d total_elapsed=%.1fs",
        len(segments), time.perf_counter() - job_start,
    )
    return _merge_consecutive(segments), language


def _transcribe_faster_whisper(audio_path: str) -> tuple[list[Segment], str]:
    from faster_whisper import WhisperModel  # type: ignore

    log.info("transcriber step=loading_fw_model model=%s device=%s", WHISPER_MODEL_SIZE, WHISPER_DEVICE)
    t0 = time.perf_counter()
    model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type="int8")
    log.info("transcriber step=fw_model_ready elapsed=%.1fs", time.perf_counter() - t0)

    log.info("transcriber step=transcribing_fw path=%s", audio_path)
    t0 = time.perf_counter()
    raw_segs, info = model.transcribe(audio_path, language=WHISPER_LANGUAGE or None)

    segments: list[Segment] = []
    for seg in raw_segs:  # generator — logs per segment for streaming feedback
        segments.append({
            "speaker": "SPEAKER_00",
            "start":   round(seg.start, 2),
            "end":     round(seg.end, 2),
            "text":    seg.text.strip(),
        })
        # log every 50 segments so you can see it's alive
        if len(segments) % 50 == 0:
            log.info("transcriber fw_progress segments=%d last_end=%.1fs",
                     len(segments), seg.end)

    log.info(
        "transcriber complete engine=faster_whisper segments=%d elapsed=%.1fs",
        len(segments), time.perf_counter() - t0,
    )
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
