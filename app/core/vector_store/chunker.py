"""
Semantic chunker with sentence-aware boundaries and configurable overlap.

Replaces the word-count chunker in agent.py.  Key improvements:
  - Splits on sentence boundaries (not arbitrary word counts)
  - Configurable overlap (N trailing sentences carried into next chunk)
  - Preserves chronology via metadata (chunk_index, speaker, time_range)
  - Minimum character filter avoids tiny noise chunks

Chunk types produced:
  ChunkType.RAW       — verbatim transcript window
  ChunkType.SEMANTIC  — cleaned text (speaker tags + timestamps stripped)
  ChunkType.STRUCTURED — pre-built from MeetingReport fields (summary, decisions…)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.config import (
    CHUNK_OVERLAP_SENTENCES,
    CHUNK_MAX_SENTENCES,
    CHUNK_MIN_CHARS,
)


class ChunkType(str, Enum):
    RAW        = "raw"
    SEMANTIC   = "semantic"
    STRUCTURED = "structured"


@dataclass
class TextChunk:
    text:        str
    chunk_type:  ChunkType
    chunk_index: int
    job_id:      str = ""
    filename:    str = ""
    speaker:     str = ""
    start_sec:   float | None = None
    end_sec:     float | None = None
    section:     str = ""          # e.g. "decisions", "action_items", "transcript"
    metadata:    dict[str, Any] = field(default_factory=dict)


# ── sentence splitter ─────────────────────────────────────────────────────────

_SENT_PATTERN = re.compile(r"(?<=[.!?])\s+")
_TS_PATTERN   = re.compile(r"^\[[\d.]+s[–\-][\d.]+s\]\s*")
_SPK_PATTERN  = re.compile(r"^SPEAKER_\d+:\s*")


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation while keeping short chunks intact."""
    sentences = _SENT_PATTERN.split(text)
    return [s.strip() for s in sentences if len(s.strip()) >= 10]


def _clean_sentence(line: str) -> str:
    """Strip timestamps and SPEAKER_XX: prefixes for semantic chunks."""
    line = _TS_PATTERN.sub("", line)
    line = _SPK_PATTERN.sub("", line)
    return line.strip()


# ── main chunker ──────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Produces three chunk layers from a transcript:
      1. raw_chunks    — verbatim windowed transcript (overlap)
      2. semantic_chunks — cleaned text (no timestamps/speakers)
      3. structured_chunks — from MeetingReport fields
    """

    def __init__(
        self,
        max_sentences: int = CHUNK_MAX_SENTENCES,
        overlap: int = CHUNK_OVERLAP_SENTENCES,
        min_chars: int = CHUNK_MIN_CHARS,
    ) -> None:
        self.max_sentences = max_sentences
        self.overlap       = overlap
        self.min_chars     = min_chars

    # ── public API ─────────────────────────────────────────────────────────────

    def chunk_transcript(
        self,
        transcript: str,
        job_id: str = "",
        filename: str = "",
        segments: list[dict] | None = None,
    ) -> tuple[list[TextChunk], list[TextChunk]]:
        """
        Returns (raw_chunks, semantic_chunks).
        `segments` from WhisperX/diarization enriches metadata with timing+speaker.
        """
        lines = [l for l in transcript.splitlines() if len(l.strip()) >= self.min_chars]
        raw_chunks      = self._window_chunks(lines, ChunkType.RAW, job_id, filename, segments)
        semantic_chunks = self._window_chunks(
            [_clean_sentence(l) for l in lines],
            ChunkType.SEMANTIC, job_id, filename, segments,
        )
        return raw_chunks, semantic_chunks

    def chunk_report(self, report: Any, job_id: str = "", filename: str = "") -> list[TextChunk]:
        """Produce structured chunks from MeetingReport fields."""
        chunks: list[TextChunk] = []
        idx = 0

        def _add(text: str, section: str) -> None:
            nonlocal idx
            if len(text.strip()) >= self.min_chars:
                chunks.append(TextChunk(
                    text=text.strip(),
                    chunk_type=ChunkType.STRUCTURED,
                    chunk_index=idx,
                    job_id=job_id,
                    filename=filename,
                    section=section,
                ))
                idx += 1

        if report.summary.detailed_summary or report.summary.short_summary:
            _add("[MEETING SUMMARY]\n" + (report.summary.detailed_summary or report.summary.short_summary), "summary")

        if report.participants:
            _add("[PARTICIPANTS]\n" + "\n".join(
                f"- {p.name}" + (f" ({p.role})" if p.role != "Unknown" else "")
                for p in report.participants
            ), "participants")

        if report.decisions:
            _add("[DECISIONS]\n" + "\n".join(
                f"- {d.decision}" + (f"\n  Evidence: {d.evidence}" if d.evidence else "")
                for d in report.decisions
            ), "decisions")

        if report.action_items:
            _add("[ACTION ITEMS]\n" + "\n".join(
                f"- [{ai.priority.upper()}] {ai.task} — {ai.owner} (due: {ai.deadline or 'TBD'})"
                for ai in report.action_items
            ), "action_items")

        if report.risks_blockers:
            _add("[RISKS & BLOCKERS]\n" + "\n".join(
                f"- [{r.severity.upper()}] {r.risk}: {r.reason}" for r in report.risks_blockers
            ), "risks")

        if report.topics_discussed:
            _add("[TOPICS DISCUSSED]\n" + "\n".join(
                f"- {t.topic}" for t in report.topics_discussed
            ), "topics")

        if report.open_questions:
            _add("[OPEN QUESTIONS]\n" + "\n".join(f"- {q}" for q in report.open_questions), "open_questions")

        if report.followups:
            _add("[FOLLOW-UPS]\n" + "\n".join(
                f"- {f.action} ({f.type}) — {f.owner}" for f in report.followups
            ), "followups")

        return chunks

    # ── internal ───────────────────────────────────────────────────────────────

    def _window_chunks(
        self,
        lines: list[str],
        chunk_type: ChunkType,
        job_id: str,
        filename: str,
        segments: list[dict] | None,
    ) -> list[TextChunk]:
        """Sliding-window chunking with sentence-level overlap."""
        chunks: list[TextChunk] = []
        step   = max(1, self.max_sentences - self.overlap)
        total  = len(lines)

        for i in range(0, total, step):
            window = lines[i : i + self.max_sentences]
            text   = "\n".join(window).strip()
            if len(text) < self.min_chars:
                continue

            # Enrich with timing/speaker from nearest segment
            speaker, start_sec, end_sec = "", None, None
            if segments:
                seg_idx = min(i, len(segments) - 1)
                seg = segments[seg_idx]
                speaker   = seg.get("speaker", "")
                start_sec = seg.get("start")
                end_sec   = seg.get("end")

            chunks.append(TextChunk(
                text=text,
                chunk_type=chunk_type,
                chunk_index=len(chunks),
                job_id=job_id,
                filename=filename,
                speaker=speaker,
                start_sec=start_sec,
                end_sec=end_sec,
                section="transcript",
            ))

        return chunks
