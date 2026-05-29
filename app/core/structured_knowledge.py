"""
Structured Knowledge — data models for NLP-extracted meeting intelligence.

This replaces raw transcript as LLM input. Instead of sending 100K+ tokens
of raw speech, we build a ~3K-token structured summary that the LLM can use
to generate summaries and strategic insights.

Token reduction: 95-99% vs full transcript LLM extraction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class SpeakerStats(BaseModel):
    name: str
    speaking_time_seconds: float = 0.0
    speaking_percentage: float = 0.0
    utterance_count: int = 0
    word_count: int = 0
    avg_sentiment: float = 0.0


class TopicSegment(BaseModel):
    title: str
    start_idx: int = 0
    end_idx: int = 0
    keywords: list[str] = Field(default_factory=list)
    speakers_involved: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    utterance_count: int = 0


class CandidateTask(BaseModel):
    text: str
    owner: Optional[str] = None
    deadline_text: Optional[str] = None
    deadline_date: Optional[str] = None  # ISO date YYYY-MM-DD
    confidence: float = 0.7
    utterance_idx: int = 0
    speaker: Optional[str] = None


class CandidateDecision(BaseModel):
    text: str
    speakers_involved: list[str] = Field(default_factory=list)
    confidence: float = 0.7
    utterance_idx: int = 0
    keywords: list[str] = Field(default_factory=list)


class ExtractedDeadline(BaseModel):
    text: str
    date: Optional[str] = None  # ISO date YYYY-MM-DD
    context: str = ""
    speaker: Optional[str] = None
    confidence: float = 0.7


class CandidateRisk(BaseModel):
    text: str
    category: str = "general"  # technical | timeline | resource | compliance | financial | dependency
    severity: str = "medium"   # critical | high | medium | low
    confidence: float = 0.7
    utterance_idx: int = 0


class SentimentSummary(BaseModel):
    overall_score: float = 0.0   # VADER compound: -1.0 to +1.0
    overall_label: str = "neutral"
    stress_level: str = "medium"
    engagement_score: float = 0.7
    by_speaker: dict[str, float] = Field(default_factory=dict)
    trend: list[float] = Field(default_factory=list)  # per-decile score


class EntityMap(BaseModel):
    people: list[str] = Field(default_factory=list)
    organizations: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    clients: list[str] = Field(default_factory=list)


class StructuredKnowledge(BaseModel):
    job_id: str
    processed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # Core stats
    duration_seconds: float = 0.0
    total_utterances: int = 0
    total_words: int = 0

    # NLP results
    speakers: list[SpeakerStats] = Field(default_factory=list)
    topics: list[TopicSegment] = Field(default_factory=list)
    entities: EntityMap = Field(default_factory=EntityMap)
    keywords: list[str] = Field(default_factory=list)

    # Extracted items
    candidate_tasks: list[CandidateTask] = Field(default_factory=list)
    candidate_decisions: list[CandidateDecision] = Field(default_factory=list)
    deadlines: list[ExtractedDeadline] = Field(default_factory=list)
    risks: list[CandidateRisk] = Field(default_factory=list)

    # Computed analytics
    sentiment: SentimentSummary = Field(default_factory=SentimentSummary)
    open_questions: list[str] = Field(default_factory=list)
    key_quotes: list[dict] = Field(default_factory=list)  # [{speaker, text}]

    def to_llm_context(self) -> str:
        """
        Compact structured representation for LLM input.
        Typical size: 2,000–5,000 tokens vs 100,000+ for raw transcript.
        The LLM uses this to write summaries and strategic insights ONLY.
        """
        parts: list[str] = []

        dur_min = int(self.duration_seconds / 60)
        parts.append(f"MEETING DURATION: {dur_min} minutes | UTTERANCES: {self.total_utterances} | WORDS: {self.total_words}")

        if self.speakers:
            speaker_lines = "  ".join(
                f"{s.name}({s.speaking_percentage:.0f}%,{s.utterance_count}u)"
                for s in self.speakers
            )
            parts.append(f"SPEAKERS: {speaker_lines}")

        if self.topics:
            parts.append("TOPICS:")
            for i, t in enumerate(self.topics, 1):
                kws = ",".join(t.keywords[:5])
                parts.append(f"  {i}. {t.title} [{int(t.duration_seconds / 60)}min] [{kws}]")

        e = self.entities
        if e.people:
            parts.append(f"PEOPLE: {', '.join(e.people[:20])}")
        if e.organizations:
            parts.append(f"ORGS: {', '.join(e.organizations[:10])}")
        if e.technologies:
            parts.append(f"TECH: {', '.join(e.technologies[:10])}")
        if e.projects:
            parts.append(f"PROJECTS: {', '.join(e.projects[:8])}")

        if self.keywords:
            parts.append(f"KEY TERMS: {', '.join(self.keywords[:20])}")

        if self.candidate_tasks:
            parts.append(f"ACTION ITEMS ({len(self.candidate_tasks)}):")
            for t in self.candidate_tasks[:20]:
                owner = f"→{t.owner}" if t.owner else ""
                due = f" due:{t.deadline_date or t.deadline_text}" if (t.deadline_date or t.deadline_text) else ""
                parts.append(f"  [{t.confidence:.0%}] {t.text[:120]}{owner}{due}")

        if self.candidate_decisions:
            parts.append(f"DECISIONS ({len(self.candidate_decisions)}):")
            for d in self.candidate_decisions[:15]:
                by = f" by:{','.join(d.speakers_involved)}" if d.speakers_involved else ""
                parts.append(f"  [{d.confidence:.0%}] {d.text[:120]}{by}")

        if self.deadlines:
            parts.append("DEADLINES:")
            for dl in self.deadlines[:10]:
                date = f" ({dl.date})" if dl.date else ""
                parts.append(f"  {dl.text}{date} — {dl.context[:60]}")

        if self.risks:
            parts.append("RISKS/BLOCKERS:")
            for r in self.risks[:10]:
                parts.append(f"  [{r.severity.upper()}][{r.category}] {r.text[:120]}")

        s = self.sentiment
        parts.append(
            f"SENTIMENT: {s.overall_label}({s.overall_score:+.2f}) | STRESS:{s.stress_level} | ENGAGEMENT:{s.engagement_score:.0%}"
        )

        if self.open_questions:
            parts.append("OPEN QUESTIONS:")
            for q in self.open_questions[:8]:
                parts.append(f"  ? {q[:100]}")

        if self.key_quotes:
            parts.append("KEY QUOTES:")
            for q in self.key_quotes[:5]:
                parts.append(f"  {q.get('speaker', '?')}: \"{q.get('text', '')[:100]}\"")

        return "\n".join(parts)
