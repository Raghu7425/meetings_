import json
import logging
import httpx
from typing import Optional
from pydantic import BaseModel, Field
from app.core.prompts import MEETING_EXTRACTION_PROMPT
from app.config import OLLAMA_BASE_URL, MEETING_LLM_MODEL, MEETING_LLM_TIMEOUT

log = logging.getLogger("extractor")


# ── Sub-schemas ────────────────────────────────────────────────────────────

class MeetingMetadataSchema(BaseModel):
    meeting_title: str = "Untitled Meeting"
    duration_minutes: Optional[int] = None
    platform: str = "Unknown"
    language: str = "en"
    transcript_confidence: float = 0.8


class SummarySchema(BaseModel):
    short_summary: str = ""
    detailed_summary: str = ""


class ParticipantSchema(BaseModel):
    name: str
    role: str = "Unknown"
    speaker_id: str = ""
    speaking_time_minutes: Optional[float] = None


class TopicSchema(BaseModel):
    topic: str
    importance: str = "medium"
    time_range: Optional[str] = None


class DecisionSchema(BaseModel):
    decision: str
    reason: Optional[str] = None
    approved_by: list[str] = Field(default_factory=list)
    confidence: float = 0.8
    evidence: str = ""


class ActionItemSchema(BaseModel):
    task_id: str = ""
    task: str
    owner: str = "Unassigned"
    deadline: Optional[str] = None
    priority: str = "medium"
    status: str = "pending"
    dependencies: list[str] = Field(default_factory=list)
    confidence: float = 0.8
    evidence: str = ""


class FollowupSchema(BaseModel):
    type: str = "email"
    owner: str = ""
    action: str
    target_person: str = ""
    deadline: Optional[str] = None


class ReminderSchema(BaseModel):
    title: str
    date_time: Optional[str] = None
    notify_before_minutes: int = 60
    related_to: str = ""


class RiskSchema(BaseModel):
    risk: str
    severity: str = "medium"
    owner: str = ""
    reason: str = ""


class SentimentSchema(BaseModel):
    overall_sentiment: str = "neutral"
    stress_level: str = "medium"
    engagement_score: float = 0.7


class TimelineItemSchema(BaseModel):
    time: str
    topic: str


class QuoteSchema(BaseModel):
    speaker: str
    quote: str


class MetricsSchema(BaseModel):
    total_action_items: int = 0
    total_decisions: int = 0
    blocked_tasks: int = 0
    high_priority_tasks: int = 0


class NextMeetingSchema(BaseModel):
    date: Optional[str] = None
    agenda: list[str] = Field(default_factory=list)


class EntitiesSchema(BaseModel):
    people: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    clients: list[str] = Field(default_factory=list)


# ── Primary report model ────────────────────────────────────────────────────

class MeetingReport(BaseModel):
    meeting_metadata: MeetingMetadataSchema = Field(default_factory=MeetingMetadataSchema)
    summary: SummarySchema = Field(default_factory=SummarySchema)
    participants: list[ParticipantSchema] = Field(default_factory=list)
    topics_discussed: list[TopicSchema] = Field(default_factory=list)
    decisions: list[DecisionSchema] = Field(default_factory=list)
    action_items: list[ActionItemSchema] = Field(default_factory=list)
    followups: list[FollowupSchema] = Field(default_factory=list)
    reminders: list[ReminderSchema] = Field(default_factory=list)
    risks_blockers: list[RiskSchema] = Field(default_factory=list)
    sentiment: SentimentSchema = Field(default_factory=SentimentSchema)
    timeline: list[TimelineItemSchema] = Field(default_factory=list)
    quotes: list[QuoteSchema] = Field(default_factory=list)
    metrics: MetricsSchema = Field(default_factory=MetricsSchema)
    next_meeting: NextMeetingSchema = Field(default_factory=NextMeetingSchema)
    open_questions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    raw_extracted_entities: EntitiesSchema = Field(default_factory=EntitiesSchema)

    @property
    def duration_minutes(self) -> Optional[int]:
        return self.meeting_metadata.duration_minutes

    @property
    def participant_count(self) -> int:
        return len(self.participants)

    @property
    def title(self) -> str:
        return self.meeting_metadata.meeting_title


# ── LLM call ───────────────────────────────────────────────────────────────

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
                    "num_predict": 4096,
                },
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


def _parse_report(raw: str) -> MeetingReport:
    raw = raw.strip()

    # Strip markdown fences if the model added them
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # Find the outermost JSON object in case there's stray text
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]

    data = json.loads(raw)
    return MeetingReport.model_validate(data)


def _empty_report(reason: str = "") -> MeetingReport:
    msg = reason or "Extraction failed — please review transcript manually."
    return MeetingReport(summary=SummarySchema(short_summary=msg, detailed_summary=msg))


async def extract_insights(transcript: str) -> MeetingReport:
    prompt = MEETING_EXTRACTION_PROMPT.format(transcript=transcript[:8000])

    for attempt in range(2):
        try:
            raw    = await _call_ollama(prompt)
            report = _parse_report(raw)
            # Auto-populate metrics if LLM left them at zero
            if report.metrics.total_action_items == 0:
                report.metrics.total_action_items = len(report.action_items)
            if report.metrics.total_decisions == 0:
                report.metrics.total_decisions = len(report.decisions)
            report.metrics.high_priority_tasks = sum(
                1 for ai in report.action_items if ai.priority == "high"
            )
            log.info(
                "[extractor] Parsed report — %d actions, %d decisions, %d participants",
                len(report.action_items), len(report.decisions), len(report.participants),
            )
            return report
        except Exception as e:
            log.warning("[extractor] Attempt %d failed: %s", attempt + 1, e)

    log.error("[extractor] Both attempts failed — returning empty report")
    return _empty_report()


def report_from_db(
    summary_text: str = "",
    decisions_list: list[str] | None = None,
    action_items_data: list[dict] | None = None,
    open_questions: list[str] | None = None,
    duration_minutes: int | None = None,
    participant_count: int | None = None,
    structured_data: dict | None = None,
) -> MeetingReport:
    """Reconstruct a MeetingReport from stored DB fields (used by resend-email)."""
    if structured_data:
        try:
            return MeetingReport.model_validate(structured_data)
        except Exception as e:
            log.warning("[extractor] Could not reconstruct from structured_data: %s", e)

    decisions = [DecisionSchema(decision=d) for d in (decisions_list or [])]
    items     = []
    for ai in (action_items_data or []):
        try:
            items.append(ActionItemSchema(**ai))
        except Exception:
            items.append(ActionItemSchema(task=str(ai)))

    return MeetingReport(
        meeting_metadata=MeetingMetadataSchema(duration_minutes=duration_minutes),
        summary=SummarySchema(short_summary=summary_text, detailed_summary=summary_text),
        participants=[ParticipantSchema(name=f"Participant {i + 1}") for i in range(participant_count or 0)],
        decisions=decisions,
        action_items=items,
        open_questions=open_questions or [],
        metrics=MetricsSchema(
            total_action_items=len(items),
            total_decisions=len(decisions),
            high_priority_tasks=sum(1 for ai in items if ai.priority == "high"),
        ),
    )
