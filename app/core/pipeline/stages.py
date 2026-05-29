"""
Pipeline stage orchestrator.

Coordinates all processing stages for a meeting upload:

  ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐
  │  Audio       │ → │  Transcribe   │ → │  Extract Insights│
  │  Extraction  │   │  (WhisperX)   │   │  (LLM → JSON)    │
  └──────────────┘   └───────────────┘   └──────────────────┘
         │                   │                     │
         ▼                   ▼                     ▼
  ffmpeg→WAV          segments+speaker       MeetingReport
                       diarization          (Pydantic schema)
                                                   │
                        ┌──────────────────────────┘
                        ▼
               ┌─────────────────┐
               │  RAG Indexing   │
               │  (Qdrant 3-col) │
               └─────────────────┘

Each stage:
  1. Updates job state in Redis (RedisJobStore)
  2. Publishes a typed event (PipelineEvent)
  3. Performs work (async / thread-offloaded for CPU-heavy)
  4. Passes results to the next stage

Incremental summarizer runs in parallel with transcription — every
SUMMARIZER_CHUNK_LINES new lines it updates the rolling summary in Redis
without blocking the transcript output stream.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
from typing import Any

from app.config import (
    MEETINGS_INPUT_DIR,
    UPLOADS_DIR,
    SUMMARIZER_CHUNK_LINES,
)
from app.core.extractor import extract_insights, extract_insights_hybrid, MeetingReport
from app.core.nlp_engine import run_nlp_pipeline
from app.core.pipeline.event_bus import PipelineEvent, publish_event
from app.core.summarizer import IncrementalSummarizer
from app.core.transcript_cleaner import clean_transcript
from app.db.redis_client import get_redis, RedisJobStore
from app.utils.logging_config import bind_context, clear_context

log = logging.getLogger("pipeline.stages")

# ── helpers ────────────────────────────────────────────────────────────────────

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


# ── Model caches — loaded once per process, reused across jobs ─────────────────
_fw_model: Any = None
_wx_model: Any = None
_wx_align: dict = {}


def _get_fw_model() -> Any:
    global _fw_model
    if _fw_model is None:
        from faster_whisper import WhisperModel  # type: ignore
        _fw_model = WhisperModel("base", device="cpu", compute_type="int8")
        log.info("faster-whisper model loaded and cached")
    return _fw_model


def _get_wx_model() -> Any:
    global _wx_model
    if _wx_model is None:
        import whisperx  # type: ignore
        _wx_model = whisperx.load_model("base", "cpu", language="en")
        log.info("whisperx model loaded and cached")
    return _wx_model


def _get_wx_align(lang: str) -> tuple:
    if lang not in _wx_align:
        import whisperx  # type: ignore
        _wx_align[lang] = whisperx.load_align_model(language_code=lang, device="cpu")
        log.info("whisperx align model loaded for lang=%s (cached)", lang)
    return _wx_align[lang]


def _extract_audio_sync(video_path: str, audio_path: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg: {result.stderr.decode()[:400]}")


def _mem_mb() -> int:
    """Current process RSS in MB (requires psutil; returns 0 if not installed)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss // 1_048_576
    except ImportError:
        return 0


def _step(ref: dict, step: str, progress: int) -> None:
    """
    Update the shared progress dict from inside the transcription thread.
    The watchdog coroutine reads this dict every 5s and publishes to Redis.
    """
    ref["current_step"] = step
    ref["progress"]     = progress
    ref["step_ts"]      = time.time()
    ref["mem_mb"]       = _mem_mb()
    log.info(
        "transcribe_step step=%s progress=%d mem_mb=%d",
        step, progress, ref["mem_mb"],
    )


def _transcribe_sync(audio_path: str, job_state_ref: dict) -> tuple[str, list[dict]]:
    """
    Full transcription pipeline with per-sub-step progress tracking.

    Sub-step → progress band:
      loading_model    11%
      loading_audio    13%   + audio duration logged
      transcribing     15-38% (updated per-batch via faster-whisper generator;
                               whisperx batch returns all at once → held at 15%)
      aligning         40%
      diarizing        50%
      assigning        57%
      building         58%
    """
    import os

    job_state_ref["step_start"] = time.time()

    # ── WhisperX path ────────────────────────────────────────────────────────────
    try:
        import whisperx  # type: ignore
        hf_token = os.getenv("HF_TOKEN", "")

        # 1. Load model (cached after first call)
        _step(job_state_ref, "loading_model", 11)
        t0 = time.perf_counter()
        model = _get_wx_model()
        log.info("transcribe model_ready elapsed=%.1fs", time.perf_counter() - t0)

        # 2. Load audio
        _step(job_state_ref, "loading_audio", 13)
        t0 = time.perf_counter()
        audio = whisperx.load_audio(audio_path)
        audio_duration_s = len(audio) / 16_000
        log.info(
            "transcribe audio_loaded duration_min=%.1f elapsed=%.1fs mem_mb=%d",
            audio_duration_s / 60, time.perf_counter() - t0, _mem_mb(),
        )

        # 3. Transcribe
        _step(job_state_ref, "transcribing", 15)
        t0 = time.perf_counter()
        result = model.transcribe(audio, batch_size=8)
        n_raw = len(result.get("segments", []))
        log.info(
            "transcribe asr_done segments=%d elapsed=%.1fs mem_mb=%d",
            n_raw, time.perf_counter() - t0, _mem_mb(),
        )

        # 4. Alignment (word-level timestamps)
        lang = result.get("language", "en")
        _step(job_state_ref, f"aligning_lang={lang}", 40)
        t0 = time.perf_counter()
        align_model, meta = _get_wx_align(lang)
        result = whisperx.align(result["segments"], align_model, meta, audio, "cpu")
        log.info(
            "transcribe alignment_done elapsed=%.1fs mem_mb=%d",
            time.perf_counter() - t0, _mem_mb(),
        )

        # 5. Diarization (speaker labels)
        if hf_token:
            _step(job_state_ref, "loading_diarizer", 50)
            t0 = time.perf_counter()
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token, device="cpu"
            )
            log.info("transcribe diarizer_loaded elapsed=%.1fs", time.perf_counter() - t0)

            _step(job_state_ref, "diarizing", 52)
            t0 = time.perf_counter()
            diarize_segments = diarize_model(audio)
            log.info(
                "transcribe diarization_done elapsed=%.1fs mem_mb=%d",
                time.perf_counter() - t0, _mem_mb(),
            )

            _step(job_state_ref, "assigning_speakers", 57)
            t0 = time.perf_counter()
            result = whisperx.assign_word_speakers(diarize_segments, result)
            log.info("transcribe speaker_assignment_done elapsed=%.1fs", time.perf_counter() - t0)
        else:
            log.warning("transcribe HF_TOKEN not set — skipping diarization, all SPEAKER_00")

        # 6. Build output
        _step(job_state_ref, "building_segments", 58)
        segs  = result.get("segments", [])
        total = float(segs[-1]["end"]) if segs else 1.0
        lines, structured = [], []
        for seg in segs:
            spk  = seg.get("speaker", "SPEAKER_00")
            s, e = seg.get("start", 0), seg.get("end", 0)
            text = seg.get("text", "").strip()
            lines.append(f"[{s:.1f}s–{e:.1f}s] {spk}: {text}")
            structured.append({"speaker": spk, "start": s, "end": e, "text": text})

        total_elapsed = time.time() - job_state_ref["step_start"]
        log.info(
            "transcribe complete engine=whisperx segments=%d total_elapsed=%.1fs",
            len(structured), total_elapsed,
        )
        return "\n".join(lines), structured

    except ImportError:
        log.warning("transcribe whisperx not installed — using faster-whisper fallback")
    except Exception as exc:
        log.error(
            "transcribe whisperx failed step=%s error=%s — falling back",
            job_state_ref.get("current_step", "unknown"), exc,
            exc_info=True,
        )

    # ── faster-whisper fallback ───────────────────────────────────────────────────
    _step(job_state_ref, "loading_fw_model", 12)
    model = _get_fw_model()

    _step(job_state_ref, "transcribing_fw", 15)
    t0    = time.perf_counter()
    gen, info = model.transcribe(
        audio_path, language="en", beam_size=5,
        vad_filter=True, word_timestamps=False,
    )
    total = float(info.duration) if info.duration else 1.0
    lines, structured = [], []

    for seg in gen:  # generator — real-time progress per segment
        text = seg.text.strip()
        lines.append(f"[{seg.start:.1f}s–{seg.end:.1f}s] {text}")
        structured.append({"speaker": "SPEAKER_00", "start": seg.start,
                           "end": seg.end, "text": text})
        # map audio position to 15-58% band
        band_progress = 15 + int((seg.end / total) * 43)
        job_state_ref["progress"]     = min(58, band_progress)
        job_state_ref["current_step"] = f"transcribing_fw seg_end={seg.end:.0f}s"
        job_state_ref["step_ts"]      = time.time()

    total_elapsed = time.time() - job_state_ref["step_start"]
    log.info(
        "transcribe complete engine=faster_whisper segments=%d total_elapsed=%.1fs",
        len(structured), total_elapsed,
    )
    return "\n".join(lines), structured


# ── Stage runner ───────────────────────────────────────────────────────────────

class PipelineRunner:
    """
    Orchestrates all pipeline stages for a single job.
    Instantiated once per job_id; holds no global mutable state.
    """

    def __init__(self, job_id: str, video_path: str, filename: str) -> None:
        self.job_id     = job_id
        self.video_path = video_path
        self.filename   = filename
        self.audio_path = video_path.rsplit(".", 1)[0] + "_audio.wav"
        self.txt_path   = os.path.join(UPLOADS_DIR, f"{job_id}.txt")
        self._progress: dict = {
            "progress":     0,
            "current_step": "queued",
            "step_ts":      time.time(),
            "step_start":   time.time(),
            "mem_mb":       0,
        }

    # ── public entry point ─────────────────────────────────────────────────────

    async def run(self) -> None:
        bind_context(job_id=self.job_id)
        try:
            await self._stage_audio()
            transcript, segments = await self._stage_transcribe()

            # Clean transcript for LLM — raw is already on disk for user download
            clean = await asyncio.to_thread(clean_transcript, transcript)
            log.info(
                "transcript cleaned job=%s raw_chars=%d clean_chars=%d reduction=%.0f%%",
                self.job_id, len(transcript), len(clean),
                100 * (1 - len(clean) / max(len(transcript), 1)),
            )

            # Kick off incremental summarizer in background using clean text
            asyncio.ensure_future(
                IncrementalSummarizer(self.job_id).process_transcript(clean)
            )

            knowledge = await self._stage_nlp(clean, segments)
            report    = await self._stage_extract(clean, segments, knowledge)
            await self._stage_rag_index(clean, segments, report)
            await self._stage_complete(clean, report)
        except Exception as exc:
            log.exception("pipeline failed job=%s", self.job_id)
            await self._mark_failed(str(exc))
        finally:
            clear_context()
            await self._cleanup()

    # ── stages ─────────────────────────────────────────────────────────────────

    async def _stage_audio(self) -> None:
        await self._update(PipelineEvent.AUDIO_EXTRACT, 5, "Extracting audio…")
        if _ffmpeg_available():
            await asyncio.to_thread(_extract_audio_sync, self.video_path, self.audio_path)
            # Remove original video to free disk space immediately
            try:
                os.unlink(self.video_path)
            except OSError:
                pass
        else:
            log.warning("ffmpeg not found — passing file directly to STT")
            self.audio_path = self.video_path

    async def _stage_transcribe(self) -> tuple[str, list[dict]]:
        await self._update(PipelineEvent.TRANSCRIBING, 10, "Transcribing audio…")

        # Watchdog: reads the shared progress dict every 5s and publishes to Redis.
        # This is the ONLY way to get live feedback from the sync thread.
        watchdog_task = asyncio.ensure_future(self._transcribe_watchdog())
        try:
            transcript, segments = await asyncio.to_thread(
                _transcribe_sync, self.audio_path, self._progress
            )
        finally:
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass

        os.makedirs(os.path.dirname(self.txt_path), exist_ok=True)
        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcript: {self.filename}\n\n{transcript}")

        await self._update(PipelineEvent.TRANSCRIBING, 60, "Transcription complete")
        return transcript, segments

    async def _transcribe_watchdog(self) -> None:
        """
        Publishes live sub-step progress to Redis every 5 seconds while
        _transcribe_sync runs in a thread.  Gives real visibility into which
        WhisperX phase is running and how long it has been stuck there.
        """
        INTERVAL = 5  # seconds between Redis updates
        while True:
            await asyncio.sleep(INTERVAL)

            step     = self._progress.get("current_step", "initializing")
            progress = self._progress.get("progress", 10)
            mem_mb   = self._progress.get("mem_mb", 0)
            step_ts  = self._progress.get("step_ts", time.time())
            step_sec = int(time.time() - step_ts)

            message = f"Transcribing [{step}] — {step_sec}s in this step"
            if mem_mb:
                message += f" | RAM {mem_mb} MB"

            # Clamp to the transcription band (10-59) so it never shows ≥60%
            display_progress = max(10, min(int(progress), 59))

            await self._update(PipelineEvent.TRANSCRIBING, display_progress, message)
            log.info(
                "watchdog job=%s step=%s progress=%d step_elapsed=%ds mem_mb=%d",
                self.job_id, step, display_progress, step_sec, mem_mb,
            )

    async def _stage_nlp(self, transcript: str, segments: list[dict]) -> "StructuredKnowledge":  # type: ignore[name-defined]
        from app.core.structured_knowledge import StructuredKnowledge
        await self._update(PipelineEvent.SUMMARIZING, 63, "NLP pre-processing…")
        try:
            knowledge = await run_nlp_pipeline(segments, self.job_id, transcript)
            log.info(
                "nlp complete job=%s tasks=%d decisions=%d risks=%d entities_people=%d",
                self.job_id,
                len(knowledge.candidate_tasks),
                len(knowledge.candidate_decisions),
                len(knowledge.risks),
                len(knowledge.entities.people),
            )
            return knowledge
        except Exception as exc:
            log.warning("NLP pipeline failed (%s) — returning empty knowledge", exc)
            return StructuredKnowledge(job_id=self.job_id)

    async def _stage_extract(
        self,
        transcript: str,
        segments: list[dict],
        knowledge: "StructuredKnowledge | None" = None,  # type: ignore[name-defined]
    ) -> MeetingReport:
        await self._update(PipelineEvent.SUMMARIZING, 65, "LLM synthesis…")
        if knowledge is not None and knowledge.total_utterances > 0:
            report = await extract_insights_hybrid(transcript, knowledge)
        else:
            report = await extract_insights(transcript)
        await self._update(PipelineEvent.SUMMARIZING, 80, "Insights extracted")
        return report

    async def _stage_rag_index(
        self,
        transcript: str,
        segments: list[dict],
        report: MeetingReport,
    ) -> None:
        await self._update(PipelineEvent.RAG_INDEXING, 82, "Building vector index…")
        try:
            from app.core.vector_store.qdrant_store import QdrantMeetingStore
            from app.core.vector_store.embedder import MultiLevelEmbedder
            from app.core.vector_store.chunker import SemanticChunker

            store   = QdrantMeetingStore()
            embedder = MultiLevelEmbedder()
            chunker  = SemanticChunker()

            await store.index_meeting(
                job_id=self.job_id,
                filename=self.filename,
                transcript=transcript,
                segments=segments,
                report=report,
                embedder=embedder,
                chunker=chunker,
            )
            log.info("qdrant index complete job=%s", self.job_id)
        except Exception as exc:
            log.warning("qdrant indexing failed (falling back to FAISS): %s", exc)
            # Preserve backward compatibility — rebuild in-memory FAISS index
            try:
                from app.api.upload import _build_rag_index
                _build_rag_index(transcript, report)
            except Exception:
                pass

        await self._update(PipelineEvent.RAG_INDEXING, 90, "Index ready")

    async def _stage_complete(self, transcript: str, report: MeetingReport) -> None:
        r = await get_redis()
        store = RedisJobStore(r)
        import json as _json

        # Redis update and knowledge-base file write run in parallel
        await asyncio.gather(
            store.update(
                self.job_id,
                status=PipelineEvent.DONE.value,
                progress="100",
                summary=report.summary.short_summary or report.summary.detailed_summary,
                structured_data=_json.dumps(report.model_dump()),
                transcript_path=self.txt_path,
            ),
            asyncio.to_thread(self._write_knowledge_base, transcript, report),
            return_exceptions=True,
        )

        # Signal voice agent to rebuild index
        try:
            from app.core.agent import invalidate_agent_index, ensure_system_initialized
            invalidate_agent_index()
            asyncio.ensure_future(ensure_system_initialized())
        except Exception:
            pass

        await publish_event(self.job_id, PipelineEvent.DONE, progress=100, message="Processing complete")
        log.info("pipeline complete job=%s", self.job_id)

    # ── helpers ────────────────────────────────────────────────────────────────

    async def _update(self, event: PipelineEvent, progress: int, message: str) -> None:
        r = await get_redis()
        store = RedisJobStore(r)
        await store.update(self.job_id, status=event.value, progress=str(progress))
        await publish_event(self.job_id, event, progress=progress, message=message)
        log.info("stage=%s progress=%d job=%s", event.value, progress, self.job_id)

    async def _mark_failed(self, error: str) -> None:
        try:
            r = await get_redis()
            store = RedisJobStore(r)
            await store.update(self.job_id, status=PipelineEvent.FAILED.value, error=error)
            await publish_event(self.job_id, PipelineEvent.FAILED, message=error)
        except Exception:
            pass

    async def _cleanup(self) -> None:
        for path in (self.audio_path, self.video_path):
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass

    def _write_knowledge_base(self, transcript: str, report: MeetingReport) -> None:
        os.makedirs(MEETINGS_INPUT_DIR, exist_ok=True)
        kb_path = os.path.join(MEETINGS_INPUT_DIR, f"{self.job_id}.txt")
        sections = [f"Meeting: {self.filename}"]

        summ = report.summary.detailed_summary or report.summary.short_summary
        if summ:
            sections.append("Summary\n" + summ)
        if report.participants:
            sections.append("Participants\n" + "\n".join(
                f"- {p.name}" + (f" ({p.role})" if p.role != "Unknown" else "")
                for p in report.participants
            ))
        if report.topics_discussed:
            sections.append("Topics Discussed\n" + "\n".join(
                f"- [{t.importance.upper()}] {t.topic}" for t in report.topics_discussed
            ))
        if report.decisions:
            sections.append("Decisions\n" + "\n".join(f"- {d.decision}" for d in report.decisions))
        if report.action_items:
            sections.append("Action Items\n" + "\n".join(
                f"- [{ai.priority.upper()}] {ai.task} — {ai.owner} (due: {ai.deadline or 'TBD'})"
                for ai in report.action_items
            ))
        if report.risks_blockers:
            sections.append("Risks\n" + "\n".join(
                f"- [{r.severity.upper()}] {r.risk}: {r.reason}" for r in report.risks_blockers
            ))
        if report.open_questions:
            sections.append("Open Questions\n" + "\n".join(f"- {q}" for q in report.open_questions))

        sections.append("Full Transcript\n" + transcript)

        with open(kb_path, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(sections))
        log.info("knowledge base written job=%s path=%s", self.job_id, kb_path)
