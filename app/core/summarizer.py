"""
Incremental rolling summarizer.

Design
──────
Traditional approach: re-send the entire transcript to the LLM each time.
  Problem: O(n) cost per chunk, exploding latency on long meetings.

This approach:
  1. Split transcript into fixed-size line windows (SUMMARIZER_CHUNK_LINES).
  2. Summarize each window independently → "chunk summaries".
  3. On each new window, combine previous rolling summary + new chunk summary
     → updated rolling summary (one short LLM call, constant cost).
  4. Store the rolling summary in Redis with a TTL.

Redis keys:
  summary:<job_id>          → JSON with rolling_summary + processed_up_to_line

The final rolling summary is returned by the GET /upload/status endpoint
and used as the RAG "summary chunk".
"""

from __future__ import annotations

import json
import logging

import httpx

from app.config import (
    OLLAMA_BASE_URL,
    MEETING_LLM_MODEL,
    MEETING_LLM_TIMEOUT,
    SUMMARIZER_CHUNK_LINES,
    SUMMARIZER_MAX_TOKENS,
    REDIS_SUMMARY_TTL,
    REDIS_SUMMARY_PREFIX,
)
from app.db.redis_client import get_redis
from app.utils.retry import retry_async

log = logging.getLogger("summarizer")

_CHUNK_SUMMARY_PROMPT = """\
Summarize the following meeting transcript excerpt in 2-3 concise sentences.
Focus only on key points, decisions, and action items.

EXCERPT:
{chunk}

SUMMARY (2-3 sentences):"""

_ROLLING_UPDATE_PROMPT = """\
You have an existing meeting summary and a new summary of the next portion.
Merge them into a single, concise rolling summary (max 4 sentences).
Preserve all key decisions, action items, and important discussions.

EXISTING SUMMARY:
{existing}

NEW PORTION SUMMARY:
{new}

UPDATED SUMMARY (max 4 sentences):"""


async def _call_ollama_compact(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=MEETING_LLM_TIMEOUT) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  MEETING_LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": SUMMARIZER_MAX_TOKENS,
                },
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()


class IncrementalSummarizer:
    """
    Maintains a rolling summary per job.  Designed to run concurrently with
    transcription — call process_transcript() as soon as the full transcript
    is available, or process_chunk() as each chunk arrives.
    """

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self._key   = f"{REDIS_SUMMARY_PREFIX}{job_id}"

    # ── public API ─────────────────────────────────────────────────────────────

    async def process_transcript(self, transcript: str) -> str:
        """
        Process the full transcript incrementally.
        Returns the final rolling summary string.
        """
        lines = [l.strip() for l in transcript.splitlines() if len(l.strip()) > 20]
        if not lines:
            return ""

        state = await self._load_state()
        start = state.get("processed_up_to_line", 0)

        if start >= len(lines):
            return state.get("rolling_summary", "")

        chunks = [
            lines[i : i + SUMMARIZER_CHUNK_LINES]
            for i in range(start, len(lines), SUMMARIZER_CHUNK_LINES)
        ]

        for i, chunk_lines in enumerate(chunks):
            chunk_text = "\n".join(chunk_lines)
            try:
                chunk_summary = await retry_async(
                    _call_ollama_compact,
                    _CHUNK_SUMMARY_PROMPT.format(chunk=chunk_text),
                    max_attempts=3,
                )
                rolling = state.get("rolling_summary", "")
                if rolling:
                    rolling = await retry_async(
                        _call_ollama_compact,
                        _ROLLING_UPDATE_PROMPT.format(existing=rolling, new=chunk_summary),
                        max_attempts=3,
                    )
                else:
                    rolling = chunk_summary

                state["rolling_summary"]      = rolling
                state["processed_up_to_line"] = start + (i + 1) * SUMMARIZER_CHUNK_LINES
                await self._save_state(state)
                log.debug(
                    "[summarizer] chunk %d/%d done job=%s",
                    i + 1, len(chunks), self.job_id,
                )
            except Exception as exc:
                log.warning("[summarizer] chunk %d failed job=%s: %s", i, self.job_id, exc)
                continue

        return state.get("rolling_summary", "")

    async def get_summary(self) -> str:
        """Return the current rolling summary (empty string if not yet computed)."""
        state = await self._load_state()
        return state.get("rolling_summary", "")

    # ── state persistence ─────────────────────────────────────────────────────

    async def _load_state(self) -> dict:
        try:
            r = await get_redis()
            raw = await r.get(self._key)
            return json.loads(raw) if raw else {}
        except Exception as exc:
            log.warning("[summarizer] load state failed job=%s: %s", self.job_id, exc)
            return {}

    async def _save_state(self, state: dict) -> None:
        try:
            r = await get_redis()
            await r.setex(self._key, REDIS_SUMMARY_TTL, json.dumps(state))
        except Exception as exc:
            log.warning("[summarizer] save state failed job=%s: %s", self.job_id, exc)
