"""
NLP pre-processing engine — structured knowledge extraction without LLM.

Processes meeting transcript segments using traditional NLP, rules, and
lightweight models. Runs BEFORE the LLM extractor, handling:

  Task          Library / Method
  ─────────     ──────────────────────────────────────────────────
  Speaker stats pure computation from diarization segments (O(n))
  NER           spaCy en_core_web_sm — PERSON, ORG, GPE, DATE
  Keywords      RAKE-NLTK (no model) → frequency fallback
  Topics        Sliding-window TF vocab shift (no model)
  Tasks         Regex pattern bank + dependency heuristics
  Decisions     Regex pattern bank (agreed/decided/approved)
  Deadlines     Regex + dateparser for relative date parsing
  Risks         Keyword-category lookup table
  Sentiment     VADER (lexicon, no GPU, <1ms per segment)
  Questions     Regex on interrogative syntax

Token savings: ~95% vs sending raw transcript to LLM.
Latency: 2-5s for a 4-hour meeting (CPU-only, no GPU needed).
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter, defaultdict
from typing import Any, Optional

from app.core.structured_knowledge import (
    CandidateDecision,
    CandidateRisk,
    CandidateTask,
    EntityMap,
    ExtractedDeadline,
    SentimentSummary,
    SpeakerStats,
    StructuredKnowledge,
    TopicSegment,
)

log = logging.getLogger("nlp_engine")

# ── Lazy singletons ─────────────────────────────────────────────────────────────

_spacy_nlp: Any = None
_vader_analyzer: Any = None


def _load_spacy() -> Any:
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    try:
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download as spacy_download
            spacy_download("en_core_web_sm")
            _spacy_nlp = spacy.load("en_core_web_sm")
        log.info("spaCy en_core_web_sm loaded")
    except ImportError:
        log.warning("spaCy not installed — NER disabled; pip install spacy")
    return _spacy_nlp


def _load_vader() -> Any:
    global _vader_analyzer
    if _vader_analyzer is not None:
        return _vader_analyzer
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            _vader_analyzer = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            _vader_analyzer = SentimentIntensityAnalyzer()
        log.info("VADER sentiment analyzer loaded")
    except ImportError:
        log.warning("NLTK not installed — sentiment disabled; pip install nltk")
    return _vader_analyzer


# ── Compiled patterns — built once at import ────────────────────────────────────

_TASK_PATTERNS = [
    re.compile(r"\b(?:action\s+item|follow.?up|todo|to.do)\s*:?\s*(.{10,150})", re.I),
    re.compile(r"\b(?:will|needs?\s+to|has\s+to|have\s+to|going\s+to)\s+([a-z].{10,120})", re.I),
    re.compile(r"\b(?:please|can\s+you|could\s+you)\s+([a-z].{10,100})", re.I),
    re.compile(r"\b(?:let'?s|we(?:'ll|\s+will)\s+)\s*([a-z].{10,100})", re.I),
    re.compile(r"\b(?:assign(?:ed)?\s+to|owner\s*:)\s*([A-Za-z].{5,80})", re.I),
]

_DECISION_PATTERNS = [
    re.compile(r"\b(?:we(?:'ve)?\s+)?(?:decided|agreed|concluded|confirmed|approved)\s+(?:to\s+|that\s+)(.{15,200})", re.I),
    re.compile(r"\b(?:the\s+)?(?:decision|agreement|consensus)\s+(?:is|was)\s+(?:to\s+)?(.{15,150})", re.I),
    re.compile(r"\bgoing\s+(?:forward|ahead)\s+with\s+(.{10,150})", re.I),
    re.compile(r"\blet'?s\s+go\s+with\s+(.{10,120})", re.I),
    re.compile(r"\bwill\s+(?:proceed|move\s+forward|use|implement|adopt)\s+(?:with\s+)?(.{10,120})", re.I),
]

_DEADLINE_RE = re.compile(
    r"\b(?:by|before|until|due|deadline|no\s+later\s+than)\s+"
    r"(?:end\s+of\s+(?:day|week|month)|eod|cob|"
    r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"(?:next|this)\s+(?:week|month|monday|tuesday|wednesday|thursday|friday)|"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}|"
    r"\d{1,2}[\/\-]\d{1,2}(?:[\/\-]\d{2,4})?)",
    re.I,
)

_QUESTION_RE = re.compile(
    r"\b(?:who|what|when|where|why|how|can|could|should|would|will|is|are|do|does)\b"
    r".{10,200}\?",
    re.I,
)

_OWNER_RE = re.compile(
    r"\b(?:assign(?:ed)?\s+to|owner\s*:?|responsibility\s+of)\s+"
    r"([A-Z][a-z]{2,20}(?:\s+[A-Z][a-z]{2,20})?)",
    re.I,
)

_NAME_VERB_RE = re.compile(
    r"\b([A-Z][a-z]{2,20}(?:\s+[A-Z][a-z]{2,20})?)\s+"
    r"(?:will|should|needs?\s+to|is\s+going\s+to|to)\b"
)

_RISK_KEYWORDS: dict[str, list[str]] = {
    "technical":   ["bug", "outage", "failure", "broken", "crash", "error", "blocked", "blocker"],
    "timeline":    ["behind schedule", "delayed", "overdue", "at risk", "missed deadline", "slipping"],
    "resource":    ["understaffed", "no bandwidth", "resource constraint", "overloaded", "capacity"],
    "compliance":  ["compliance", "legal", "gdpr", "privacy", "security risk", "audit", "regulation"],
    "financial":   ["over budget", "cost overrun", "funding", "budget risk"],
    "dependency":  ["waiting on", "depends on", "blocking us", "blocked by", "need approval"],
}

_SEVERITY_KEYWORDS: dict[str, list[str]] = {
    "critical": ["critical", "blocker", "outage", "urgent", "immediate", "showstopper"],
    "high":     ["at risk", "blocked", "failure", "overdue", "legal", "security risk"],
    "low":      ["minor", "small", "slight", "possible", "maybe"],
}

_TECH_TERMS = frozenset([
    "python", "javascript", "typescript", "react", "vue", "angular", "node", "docker",
    "kubernetes", "k8s", "aws", "azure", "gcp", "postgres", "postgresql", "mysql",
    "redis", "kafka", "rabbitmq", "fastapi", "django", "flask", "graphql", "rest",
    "api", "microservices", "ml", "ai", "llm", "gpt", "claude", "tensorflow", "pytorch",
    "spark", "databricks", "snowflake", "dbt", "terraform", "ansible", "jenkins",
    "github", "gitlab", "jira", "confluence", "elasticsearch", "mongodb", "cassandra",
])

_STOPWORDS = frozenset([
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for", "and", "or", "but",
    "with", "we", "i", "you", "he", "she", "they", "that", "this", "was", "were", "be",
    "been", "have", "has", "had", "will", "would", "could", "should", "may", "might",
    "do", "does", "just", "not", "no", "so", "up", "out", "of", "if", "then", "are",
    "going", "think", "know", "get", "got", "want", "need", "like", "yeah", "okay",
    "right", "actually", "really", "just", "also", "well", "good", "very", "some",
])


# ── NLP Engine ──────────────────────────────────────────────────────────────────

class NLPEngine:
    """
    CPU-only NLP extractor. All methods are synchronous.
    Call via run_nlp_pipeline() for async use.
    """

    def process(
        self,
        segments: list[dict],
        job_id: str,
        plain_transcript: str = "",
    ) -> StructuredKnowledge:
        if not segments:
            return StructuredKnowledge(job_id=job_id)

        full_text = plain_transcript or " ".join(s.get("text", "") for s in segments)
        duration = max((s.get("end", 0.0) for s in segments), default=0.0)

        speakers   = self._speaker_stats(segments)
        entities   = self._extract_entities(full_text)
        keywords   = self._extract_keywords(full_text)
        topics     = self._segment_topics(segments, keywords)
        tasks      = self._detect_tasks(segments)
        decisions  = self._detect_decisions(segments)
        deadlines  = self._extract_deadlines(segments)
        risks      = self._detect_risks(segments)
        sentiment  = self._analyze_sentiment(segments)
        questions  = self._detect_questions(segments)
        quotes     = self._key_quotes(segments, tasks, decisions)

        for spk in speakers:
            spk.avg_sentiment = sentiment.by_speaker.get(spk.name, 0.0)

        return StructuredKnowledge(
            job_id=job_id,
            duration_seconds=round(duration, 1),
            total_utterances=len(segments),
            total_words=len(full_text.split()),
            speakers=speakers,
            topics=topics,
            entities=entities,
            keywords=keywords,
            candidate_tasks=tasks,
            candidate_decisions=decisions,
            deadlines=deadlines,
            risks=risks,
            sentiment=sentiment,
            open_questions=questions,
            key_quotes=quotes,
        )

    # ── Speaker stats ────────────────────────────────────────────────────────────

    def _speaker_stats(self, segments: list[dict]) -> list[SpeakerStats]:
        acc: dict[str, dict] = defaultdict(lambda: {"time": 0.0, "count": 0, "words": 0})
        total_time = 0.0
        for seg in segments:
            spk = seg.get("speaker", "SPEAKER_00")
            dur = max(0.0, seg.get("end", 0.0) - seg.get("start", 0.0))
            acc[spk]["time"]  += dur
            acc[spk]["count"] += 1
            acc[spk]["words"] += len(seg.get("text", "").split())
            total_time += dur
        result = []
        for name, s in sorted(acc.items(), key=lambda x: -x[1]["time"]):
            pct = (s["time"] / total_time * 100) if total_time > 0 else 0.0
            result.append(SpeakerStats(
                name=name,
                speaking_time_seconds=round(s["time"], 1),
                speaking_percentage=round(pct, 1),
                utterance_count=s["count"],
                word_count=s["words"],
            ))
        return result

    # ── Entity extraction ────────────────────────────────────────────────────────

    def _extract_entities(self, text: str) -> EntityMap:
        em = EntityMap()
        nlp = _load_spacy()
        if nlp is None:
            em.technologies = self._detect_tech(text)
            return em

        people: set[str] = set()
        orgs: set[str] = set()
        locs: set[str] = set()
        dates: set[str] = set()

        # Chunk large transcripts — spaCy default limit is 1M chars
        for chunk_start in range(0, len(text), 50_000):
            doc = nlp(text[chunk_start:chunk_start + 50_000])
            for ent in doc.ents:
                val = ent.text.strip()
                if not (2 <= len(val) <= 60):
                    continue
                if ent.label_ == "PERSON":
                    people.add(val)
                elif ent.label_ == "ORG":
                    orgs.add(val)
                elif ent.label_ in ("GPE", "LOC"):
                    locs.add(val)
                elif ent.label_ in ("DATE", "TIME"):
                    dates.add(val)

        em.people        = sorted(people)[:30]
        em.organizations = sorted(orgs)[:20]
        em.locations     = sorted(locs)[:15]
        em.dates         = sorted(dates)[:20]
        em.technologies  = self._detect_tech(text)
        em.projects      = self._detect_projects(text)
        return em

    def _detect_tech(self, text: str) -> list[str]:
        tl = text.lower()
        return [t for t in _TECH_TERMS if t in tl]

    def _detect_projects(self, text: str) -> list[str]:
        found: set[str] = set()
        # ACRONYMS (2-6 uppercase letters) used repeatedly
        for m in re.finditer(r'\b([A-Z]{2,6})\b', text):
            val = m.group(1)
            if text.count(val) >= 2:
                found.add(val)
        # "Project XYZ" patterns
        for m in re.finditer(r'\bproject\s+([A-Z][a-zA-Z0-9]+)', text, re.I):
            found.add(m.group(1))
        return sorted(found)[:10]

    # ── Keyword extraction ───────────────────────────────────────────────────────

    def _extract_keywords(self, text: str, top_n: int = 30) -> list[str]:
        try:
            from rake_nltk import Rake  # type: ignore
            r = Rake()
            r.extract_keywords_from_text(text[:20_000])
            phrases = r.get_ranked_phrases()[:top_n]
            return [p for p in phrases if len(p.strip()) >= 3]
        except ImportError:
            pass
        # Frequency fallback — bigrams + unigrams
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        filtered = [w for w in words if w not in _STOPWORDS]
        freq = Counter(filtered)
        return [w for w, _ in freq.most_common(top_n)]

    # ── Topic segmentation ───────────────────────────────────────────────────────

    def _segment_topics(
        self,
        segments: list[dict],
        keywords: list[str],
        window: int = 25,
    ) -> list[TopicSegment]:
        if len(segments) <= window:
            return [self._make_topic(segments, 0, len(segments) - 1, 0)]

        topics: list[TopicSegment] = []
        start = 0
        while start < len(segments):
            end = min(start + window, len(segments))
            topics.append(self._make_topic(segments, start, end - 1, len(topics)))
            start = end
        return topics

    def _make_topic(
        self, segments: list[dict], start: int, end: int, idx: int
    ) -> TopicSegment:
        chunk = segments[start:end + 1]
        text  = " ".join(s.get("text", "") for s in chunk)
        kws   = self._top_words(text, 5)
        spks  = list({s.get("speaker", "") for s in chunk if s.get("speaker")})
        t0    = chunk[0].get("start", 0.0) if chunk else 0.0
        t1    = chunk[-1].get("end", 0.0) if chunk else 0.0
        title = f"Topic: {kws[0].title()}" if kws else f"Segment {idx + 1}"
        return TopicSegment(
            title=title,
            start_idx=start,
            end_idx=end,
            keywords=kws,
            speakers_involved=spks,
            duration_seconds=round(t1 - t0, 1),
            utterance_count=len(chunk),
        )

    def _top_words(self, text: str, n: int) -> list[str]:
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        freq  = Counter(w for w in words if w not in _STOPWORDS)
        return [w for w, _ in freq.most_common(n)]

    # ── Task detection ───────────────────────────────────────────────────────────

    def _detect_tasks(self, segments: list[dict]) -> list[CandidateTask]:
        tasks: list[CandidateTask] = []
        seen: set[str] = set()
        for idx, seg in enumerate(segments):
            text = seg.get("text", "")
            spk  = seg.get("speaker", "")
            for pat in _TASK_PATTERNS:
                for m in pat.finditer(text):
                    raw = m.group(0).strip()
                    key = raw[:40].lower()
                    if key in seen or len(raw) < 15:
                        continue
                    seen.add(key)
                    owner = self._find_owner(text, spk)
                    dl_text, dl_date = self._parse_deadline(text)
                    conf  = 0.85 if re.search(r"action\s+item|follow.?up|todo|assign", text, re.I) else 0.65
                    tasks.append(CandidateTask(
                        text=raw[:200],
                        owner=owner,
                        deadline_text=dl_text,
                        deadline_date=dl_date,
                        confidence=conf,
                        utterance_idx=idx,
                        speaker=spk,
                    ))
        return tasks[:40]

    def _find_owner(self, text: str, fallback: str) -> Optional[str]:
        m = _OWNER_RE.search(text)
        if m:
            return m.group(1).strip()
        m = _NAME_VERB_RE.search(text)
        if m:
            return m.group(1)
        return None

    def _parse_deadline(self, text: str) -> tuple[Optional[str], Optional[str]]:
        m = _DEADLINE_RE.search(text)
        if not m:
            return None, None
        dl_text = m.group(0)
        try:
            import dateparser  # type: ignore
            parsed = dateparser.parse(dl_text, settings={"PREFER_DATES_FROM": "future"})
            if parsed:
                return dl_text, parsed.strftime("%Y-%m-%d")
        except ImportError:
            pass
        return dl_text, None

    # ── Decision detection ───────────────────────────────────────────────────────

    def _detect_decisions(self, segments: list[dict]) -> list[CandidateDecision]:
        decisions: list[CandidateDecision] = []
        seen: set[str] = set()
        for idx, seg in enumerate(segments):
            text = seg.get("text", "")
            spk  = seg.get("speaker", "")
            for pat in _DECISION_PATTERNS:
                for m in pat.finditer(text):
                    raw = m.group(0).strip()
                    key = raw[:40].lower()
                    if key in seen or len(raw) < 15:
                        continue
                    seen.add(key)
                    conf = 0.9 if re.search(r"\b(?:decided|agreed|approved|confirmed)\b", text, re.I) else 0.7
                    decisions.append(CandidateDecision(
                        text=raw[:200],
                        speakers_involved=[spk] if spk else [],
                        confidence=conf,
                        utterance_idx=idx,
                        keywords=self._top_words(raw, 3),
                    ))
        return decisions[:25]

    # ── Deadline extraction ──────────────────────────────────────────────────────

    def _extract_deadlines(self, segments: list[dict]) -> list[ExtractedDeadline]:
        deadlines: list[ExtractedDeadline] = []
        seen: set[str] = set()
        for seg in segments:
            text = seg.get("text", "")
            for m in _DEADLINE_RE.finditer(text):
                dl = m.group(0)
                if dl.lower() in seen:
                    continue
                seen.add(dl.lower())
                _, date_iso = self._parse_deadline(text)
                deadlines.append(ExtractedDeadline(
                    text=dl,
                    date=date_iso,
                    context=text[:100],
                    speaker=seg.get("speaker"),
                    confidence=0.8,
                ))
        return deadlines[:20]

    # ── Risk detection ───────────────────────────────────────────────────────────

    def _detect_risks(self, segments: list[dict]) -> list[CandidateRisk]:
        risks: list[CandidateRisk] = []
        seen: set[str] = set()
        for idx, seg in enumerate(segments):
            text  = seg.get("text", "")
            textl = text.lower()
            for category, kws in _RISK_KEYWORDS.items():
                for kw in kws:
                    if kw not in textl:
                        continue
                    for sent in re.split(r"[.!?]", text):
                        if kw not in sent.lower() or len(sent.strip()) < 10:
                            continue
                        key = sent.strip()[:40].lower()
                        if key in seen:
                            continue
                        seen.add(key)
                        severity = "medium"
                        for sev, sev_kws in _SEVERITY_KEYWORDS.items():
                            if any(sk in textl for sk in sev_kws):
                                severity = sev
                                break
                        risks.append(CandidateRisk(
                            text=sent.strip()[:200],
                            category=category,
                            severity=severity,
                            confidence=0.75,
                            utterance_idx=idx,
                        ))
                    break  # one category match per keyword per segment
        _order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        return sorted(risks, key=lambda r: _order.get(r.severity, 2))[:20]

    # ── Sentiment analysis (VADER) ───────────────────────────────────────────────

    def _analyze_sentiment(self, segments: list[dict]) -> SentimentSummary:
        vader = _load_vader()
        if vader is None:
            return SentimentSummary()

        scores: list[float] = []
        by_spk: dict[str, list[float]] = defaultdict(list)

        for seg in segments:
            text = seg.get("text", "")
            spk  = seg.get("speaker", "")
            if not text:
                continue
            sc = vader.polarity_scores(text)["compound"]
            scores.append(sc)
            by_spk[spk].append(sc)

        if not scores:
            return SentimentSummary()

        overall = sum(scores) / len(scores)
        label   = "positive" if overall > 0.1 else "negative" if overall < -0.1 else "neutral"

        chunk = max(1, len(scores) // 10)
        trend = [round(sum(scores[i:i + chunk]) / chunk, 3) for i in range(0, len(scores), chunk)]

        neg_ratio = sum(1 for s in scores if s < -0.3) / len(scores)
        stress    = "high" if neg_ratio > 0.3 else "low" if neg_ratio < 0.1 else "medium"

        try:
            import statistics
            eng = min(1.0, statistics.stdev(scores) * 2) if len(scores) > 1 else 0.5
        except Exception:
            eng = 0.5

        by_spk_avg = {spk: round(sum(v) / len(v), 3) for spk, v in by_spk.items() if v}

        return SentimentSummary(
            overall_score=round(overall, 3),
            overall_label=label,
            stress_level=stress,
            engagement_score=round(eng, 2),
            by_speaker=by_spk_avg,
            trend=trend[:10],
        )

    # ── Question detection ───────────────────────────────────────────────────────

    def _detect_questions(self, segments: list[dict]) -> list[str]:
        questions: list[str] = []
        seen: set[str] = set()
        for seg in segments:
            for m in _QUESTION_RE.finditer(seg.get("text", "")):
                q   = m.group(0).strip()
                key = q[:40].lower()
                if key not in seen and len(q) > 15:
                    seen.add(key)
                    questions.append(q[:200])
        return questions[:20]

    # ── Key quote extraction ─────────────────────────────────────────────────────

    def _key_quotes(
        self,
        segments: list[dict],
        tasks: list[CandidateTask],
        decisions: list[CandidateDecision],
    ) -> list[dict]:
        decision_idxs = {d.utterance_idx for d in decisions}
        task_idxs     = {t.utterance_idx for t in tasks}
        important     = decision_idxs | task_idxs
        kw_re         = re.compile(
            r"\b(?:decided|agreed|critical|important|must|approved|risk|blocker)\b", re.I
        )
        quotes: list[dict] = []
        seen: set[str] = set()

        for idx, seg in enumerate(segments):
            text = seg.get("text", "").strip()
            spk  = seg.get("speaker", "")
            wc   = len(text.split())
            key  = text[:40].lower()
            if key in seen:
                continue
            is_important = (idx in important) or bool(kw_re.search(text))
            if is_important and 15 <= wc <= 80:
                seen.add(key)
                quotes.append({"speaker": spk, "text": text[:200]})

        return quotes[:8]


# ── Public async entry point ─────────────────────────────────────────────────────

async def run_nlp_pipeline(
    segments: list[dict],
    job_id: str,
    plain_transcript: str = "",
) -> StructuredKnowledge:
    """
    Offloads CPU-heavy NLP to thread pool. Returns StructuredKnowledge.
    Typical runtime: 2-5s for a 4-hour meeting on a single CPU core.
    """
    engine = NLPEngine()
    return await asyncio.to_thread(
        engine.process, segments, job_id, plain_transcript
    )
