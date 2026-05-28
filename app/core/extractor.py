import json
import logging
import asyncio
import httpx
from datetime import date
from pydantic import BaseModel, Field
from typing import Optional
from app.core.prompts import MEETING_EXTRACTION_PROMPT
from app.config import OLLAMA_BASE_URL, MEETING_LLM_MODEL, MEETING_LLM_TIMEOUT

log = logging.getLogger("extractor")


class ActionItemSchema(BaseModel):
    task:     str
    owner:    str = "Unassigned"
    deadline: Optional[str] = None
    priority: str = "medium"


class MeetingReport(BaseModel):
    summary:           str = ""
    decisions:         list[str] = Field(default_factory=list)
    action_items:      list[ActionItemSchema] = Field(default_factory=list)
    open_questions:    list[str] = Field(default_factory=list)
    duration_minutes:  Optional[int] = None
    participant_count: Optional[int] = None


async def _call_ollama(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=MEETING_LLM_TIMEOUT) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":      MEETING_LLM_MODEL,
                "prompt":     prompt,
                "stream":     False,
                "keep_alive": -1,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 2048,
                },
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


def _parse_report(raw: str) -> MeetingReport:
    raw = raw.strip()

    # Strip markdown code fences if the model included them anyway
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    data = json.loads(raw)
    return MeetingReport(**data)


async def extract_insights(transcript: str) -> MeetingReport:
    prompt = MEETING_EXTRACTION_PROMPT.format(transcript=transcript)

    for attempt in range(2):
        try:
            raw = await _call_ollama(prompt)
            report = _parse_report(raw)
            log.info(
                f"[extractor] Parsed report — {len(report.action_items)} actions, "
                f"{len(report.decisions)} decisions"
            )
            return report
        except (json.JSONDecodeError, Exception) as e:
            log.warning(f"[extractor] Attempt {attempt + 1} failed: {e}")
            if attempt == 1:
                log.error("[extractor] Both attempts failed — returning empty report")
                return MeetingReport(summary="Extraction failed — please review transcript manually.")

    return MeetingReport()
