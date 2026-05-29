"""
Conversational Intelligence API.

Exposes the graph-powered and JSONB-powered analytics layer.
Endpoints degrade gracefully: if Memgraph is disabled, most endpoints
fall back to PostgreSQL JSONB queries so they always return useful data.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

log = logging.getLogger("api.intelligence")
router = APIRouter(prefix="/intelligence", tags=["intelligence"])


# ── request / response schemas ────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    meeting_id: str | None = None
    top_k: int = 5


# ── topic intelligence ────────────────────────────────────────────────────────

@router.get("/topics")
async def list_topics():
    """
    All discovered topics with frequency and keywords.
    Merges BERTopic model topics with JSONB-mined topics.
    """
    graph_task = _safe(
        lambda: __import__("app.core.graph.queries", fromlist=["all_topics"]).all_topics()
    )
    pg_task = _safe(
        lambda: __import__(
            "app.core.intelligence.analytics", fromlist=["get_topic_analytics"]
        ).get_topic_analytics()
    )
    bertopic_task = _safe(
        lambda: __import__(
            "app.core.nlp.topic_engine", fromlist=["get_topic_engine"]
        ).get_topic_engine().get_all_topics_async()
    )

    graph_topics, pg_topics, bertopic_topics = await asyncio.gather(
        graph_task, pg_task, bertopic_task, return_exceptions=True
    )

    return {
        "graph_topics":    graph_topics if not isinstance(graph_topics, Exception) else [],
        "pg_topics":       pg_topics if not isinstance(pg_topics, Exception) else [],
        "bertopic_topics": bertopic_topics if not isinstance(bertopic_topics, Exception) else [],
    }


@router.get("/topics/{topic_name}/history")
async def topic_history(topic_name: str):
    """
    All meetings where this topic was discussed, plus who tends to own it.
    """
    from app.core.graph.queries import topic_history as gh, topic_owners as go
    history, owners = await asyncio.gather(gh(topic_name), go(topic_name))
    return {"topic": topic_name, "meetings": history, "typical_owners": owners}


@router.get("/topics/{topic_name}/retention")
async def topic_retention(topic_name: str, limit: int = Query(10, ge=1, le=50)):
    """Decisions and action items historically related to a topic keyword."""
    from app.core.intelligence.analytics import get_knowledge_retention
    return await get_knowledge_retention(topic_name, limit)


# ── speaker intelligence ──────────────────────────────────────────────────────

@router.get("/speakers")
async def all_speakers():
    """Aggregated analytics for every speaker across all processed meetings."""
    from app.core.intelligence.analytics import get_speaker_analytics
    return await get_speaker_analytics()


@router.get("/speakers/{speaker_name}")
async def speaker_profile(speaker_name: str):
    """
    Full profile for one speaker: meeting count, speaking %, top topics,
    action-item ownership rate — merged from graph + PostgreSQL.
    """
    graph_task = _safe(
        lambda: __import__(
            "app.core.graph.queries", fromlist=["speaker_profile"]
        ).speaker_profile(speaker_name)
    )
    pg_task = _safe(
        lambda: __import__(
            "app.core.intelligence.analytics", fromlist=["get_speaker_analytics"]
        ).get_speaker_analytics(speaker_name)
    )
    graph_data, pg_data = await asyncio.gather(graph_task, pg_task)
    return {
        "name":        speaker_name,
        "graph":       graph_data if not isinstance(graph_data, Exception) else {},
        "meetings":    pg_data if not isinstance(pg_data, Exception) else [],
    }


# ── meeting intelligence ──────────────────────────────────────────────────────

@router.get("/meetings/{meeting_id}/context")
async def meeting_context(meeting_id: str):
    """
    Full graph context for a meeting: participants, topics, decisions, risks,
    similar meetings.
    """
    from app.core.graph.queries import meeting_context as gc
    from app.core.intelligence.analytics import get_interaction_analysis
    ctx, interaction = await asyncio.gather(
        gc(meeting_id),
        get_interaction_analysis(meeting_id),
    )
    return {**ctx, "interaction_analysis": interaction}


@router.get("/meetings/{meeting_id}/summaries")
async def meeting_summaries(meeting_id: str):
    """
    Returns the three Sumy extractive summaries stored in structured_data.
    Falls back to recomputing on-the-fly if not stored.
    """
    from app.db.database import AsyncSessionLocal
    from sqlalchemy import text

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text(
                "SELECT structured_data->'extractive_summaries' AS s, "
                "       transcript_path "
                "FROM meetings WHERE id = :mid"
            ),
            {"mid": meeting_id},
        )
        row = result.one_or_none()
        if row is None:
            raise HTTPException(404, "Meeting not found")
        if row.s:
            return row.s

    # Not stored — recompute from transcript
    from app.core.storage import download_file
    from app.core.nlp.summarizer import multi_summary_async

    try:
        text_content = await download_file(row.transcript_path)
        return await multi_summary_async(text_content.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise HTTPException(500, f"Could not compute summaries: {exc}") from exc


# ── cross-meeting intelligence ────────────────────────────────────────────────

@router.get("/recurring-issues")
async def recurring_issues(
    min_frequency: int = Query(2, ge=1, le=20, description="Min meeting count"),
):
    """Risks / blockers that appeared in multiple meetings."""
    from app.core.graph.queries import recurring_issues as ri
    return await ri(min_frequency)


@router.get("/decisions")
async def historical_decisions(
    limit: int = Query(50, ge=1, le=200),
):
    """All decisions logged across meetings, newest first."""
    from app.core.graph.queries import historical_decisions as hd
    return await hd(limit)


@router.get("/trends")
async def meeting_trends(days: int = Query(90, ge=7, le=365)):
    """
    Weekly meeting trends: volume, average duration, sentiment.
    Merges graph and PostgreSQL data.
    """
    from app.core.graph.queries import meeting_trends as gt
    from app.core.intelligence.analytics import get_meeting_trends_pg

    graph_trends, pg_trends = await asyncio.gather(
        gt(days),
        get_meeting_trends_pg(days),
    )
    return {
        "graph_trends": graph_trends if not isinstance(graph_trends, Exception) else [],
        **pg_trends if isinstance(pg_trends, dict) else {},
    }


# ── semantic search ───────────────────────────────────────────────────────────

@router.post("/search")
async def semantic_search(req: SearchRequest):
    """
    Typo-tolerant semantic search across all meeting chunks.
    Uses SentenceTransformers embeddings + Qdrant hybrid retrieval.
    Optionally restricted to a single meeting_id.
    """
    from app.core.vector_store.embedder import MultiLevelEmbedder
    from app.core.vector_store.qdrant_store import QdrantMeetingStore

    embedder = MultiLevelEmbedder()
    query_vec = await embedder.embed_for_query(req.query)
    store = QdrantMeetingStore()

    if not store.available:
        raise HTTPException(503, "Vector store unavailable")

    results = await store.search_hybrid(req.query, query_vec[0], req.top_k)
    return {"query": req.query, "results": results}


# ── helpers ───────────────────────────────────────────────────────────────────

async def _safe(factory):
    """Invoke an async factory safely, propagating exceptions for gather()."""
    try:
        return await factory()
    except Exception as exc:
        log.debug("Intelligence call failed: %s", exc)
        raise
