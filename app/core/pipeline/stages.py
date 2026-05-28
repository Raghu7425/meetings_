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
from typing import Any

from app.config import (
    MEETINGS_INPUT_DIR,
    UPLOADS_DIR,
    SUMMARIZER_CHUNK_LINES,
)
from app.core.extractor import extract_insights, MeetingReport
from app.core.pipeline.event_bus import PipelineEvent, publish_event
from app.core.summarizer import IncrementalSummarizer
from app.db.redis_client import get_redis, RedisJobStore
from app.utils.logging_config import bind_context, clear_context

log = logging.getLogger("pipeline.stages")

# ── helpers ────────────────────────────────────────────────────────────────────

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _extract_audio_sync(video_path: str, audio_path: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg: {result.stderr.decode()[:400]}")


def _transcribe_sync(audio_path: str, job_state_ref: dict) -> tuple[str, list[dict]]:
    """
    Returns (plain_transcript_str, segments_list).
    segments_list: [{speaker, start, end, text}, ...]
    job_state_ref["progress"] is mutated in-place for progress updates.
    """
    import os

    try:
        import whisperx
        hf_token = os.getenv("HF_TOKEN", "")
        model = whisperx.load_model("base", "cpu", language="en")
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=16)
        lang = result.get("language", "en")
        align_model, meta = whisperx.load_align_model(language_code=lang, device="cpu")
        result = whisperx.align(result["segments"], align_model, meta, audio, "cpu")
        if hf_token:
            diarize = whisperx.DiarizationPipeline(use_auth_token=hf_token, device="cpu")
            result = whisperx.assign_word_speakers(diarize(audio), result)

        segs = result.get("segments", [])
        total = float(segs[-1]["end"]) if segs else 1.0
        lines, structured = [], []
        for seg in segs:
            spk = seg.get("speaker", "SPEAKER_00")
            s, e = seg.get("start", 0), seg.get("end", 0)
            text = seg.get("text", "").strip()
            lines.append(f"[{s:.1f}s–{e:.1f}s] {spk}: {text}")
            structured.append({"speaker": spk, "start": s, "end": e, "text": text})
            job_state_ref["progress"] = min(95, int((e / total) * 100))

        return "\n".join(lines), structured

    except ImportError:
        pass
    except Exception as exc:
        log.warning("whisperx failed (%s), falling back to faster-whisper", exc)

    from faster_whisper import WhisperModel
    model = WhisperModel("base", device="cpu", compute_type="int8")
    gen, info = model.transcribe(audio_path, language="en", beam_size=5,
                                  vad_filter=True, word_timestamps=False)
    total = float(info.duration) if info.duration else 1.0
    lines, structured = [], []
    for seg in gen:
        text = seg.text.strip()
        lines.append(f"[{seg.start:.1f}s–{seg.end:.1f}s] {text}")
        structured.append({"speaker": "SPEAKER_00", "start": seg.start, "end": seg.end, "text": text})
        job_state_ref["progress"] = min(95, int((seg.end / total) * 100))
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
        self._progress  = {"progress": 0}  # shared ref for thread updates

    # ── public entry point ─────────────────────────────────────────────────────

    async def run(self) -> None:
        bind_context(job_id=self.job_id)
        try:
            await self._stage_audio()
            transcript, segments = await self._stage_transcribe()
            report = await self._stage_extract(transcript)
            await self._stage_rag_index(transcript, segments, report)
            await self._stage_complete(transcript, report)
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

        # Run incremental summarizer in a background task while transcription runs
        summarizer = IncrementalSummarizer(self.job_id)

        # CPU-heavy — offload to thread
        transcript, segments = await asyncio.to_thread(
            _transcribe_sync, self.audio_path, self._progress
        )

        # Write plain-text transcript to disk
        os.makedirs(os.path.dirname(self.txt_path), exist_ok=True)
        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcript: {self.filename}\n\n{transcript}")

        # Kick off incremental summary in background (non-blocking)
        asyncio.ensure_future(summarizer.process_transcript(transcript))

        await self._update(PipelineEvent.TRANSCRIBING, 60, "Transcription complete")
        return transcript, segments

    async def _stage_extract(self, transcript: str) -> MeetingReport:
        await self._update(PipelineEvent.SUMMARIZING, 65, "Extracting meeting intelligence…")
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
        await store.update(
            self.job_id,
            status=PipelineEvent.DONE.value,
            progress="100",
            summary=report.summary.short_summary or report.summary.detailed_summary,
            structured_data=_json.dumps(report.model_dump()),
            transcript_path=self.txt_path,
        )

        # Persist knowledge-base file for voice agent RAG
        await asyncio.to_thread(self._write_knowledge_base, transcript, report)

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
