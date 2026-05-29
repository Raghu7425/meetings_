"""
Post-pipeline Memgraph sync.

Called as a fire-and-forget asyncio.create_task() after DB persist.
Never raises — all errors are logged and swallowed so the main pipeline
is never blocked or broken by graph failures.

Also computes meeting similarity via Qdrant and creates SIMILAR_TO edges.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.structured_knowledge import StructuredKnowledge

log = logging.getLogger("graph.sync")


async def sync_meeting_to_graph(
    meeting_id: str,
    meta: dict,
    knowledge: "StructuredKnowledge | None",
    report,  # MeetingReport — avoid circular import at module level
) -> None:
    """
    Sync one meeting's extracted knowledge to Memgraph.

    Args:
        meeting_id: PostgreSQL UUID (used as the graph node ID).
        meta:       Dict with call_id, title, date, duration_minutes, summary.
        knowledge:  StructuredKnowledge from NLP pipeline.
        report:     MeetingReport from extractor (can be None on NLP-only path).
    """
    try:
        from app.config import MEMGRAPH_ENABLED
        if not MEMGRAPH_ENABLED:
            return

        from app.core.graph.client import get_graph_client
        from app.core.nlp.entity_resolver import get_entity_resolver

        client = get_graph_client()
        resolver = get_entity_resolver()

        # 1. Meeting node
        sentiment_label = "neutral"
        engagement = 0.5
        if knowledge:
            sentiment_label = knowledge.sentiment.overall_label
            engagement = knowledge.sentiment.engagement_score

        await client.upsert_meeting({
            "id":            meeting_id,
            "call_id":       meta.get("call_id", ""),
            "title":         meta.get("title", ""),
            "date":          str(meta.get("date", "")),
            "duration_min":  meta.get("duration_minutes", 0),
            "summary":       (meta.get("summary") or "")[:1000],
            "sentiment_label": sentiment_label,
            "engagement":    engagement,
            "platform":      "teams",
        })

        if not knowledge:
            return

        # 2. Speakers
        tasks = [
            client.link_participant(
                meeting_id,
                resolver.resolve(spk.name),
                {
                    "speaking_time":  spk.speaking_time_seconds,
                    "speaking_pct":   spk.speaking_percentage,
                    "utterance_count": spk.utterance_count,
                    "sentiment_score": spk.avg_sentiment,
                },
            )
            for spk in knowledge.speakers
        ]

        # 3. Topics
        tasks += [
            client.upsert_topic(
                topic.title.removeprefix("Topic: ").strip() or f"Segment {i}",
                topic.keywords,
                meeting_id,
                topic.duration_seconds,
            )
            for i, topic in enumerate(knowledge.topics)
        ]

        # 4. Decisions
        tasks += [
            client.add_decision(
                meeting_id,
                dec.text,
                dec.confidence,
                resolver.resolve(dec.speakers_involved[0]) if dec.speakers_involved else None,
            )
            for dec in knowledge.candidate_decisions
        ]

        # 5. Action items
        tasks += [
            client.add_action_item(
                meeting_id,
                task.text,
                resolver.resolve(task.owner) if task.owner else None,
                "medium",
                task.deadline_date,
            )
            for task in knowledge.candidate_tasks
        ]

        # 6. Risks
        tasks += [
            client.add_risk(meeting_id, r.text, r.category, r.severity)
            for r in knowledge.risks
        ]

        # 7. Organizations
        tasks += [
            client.link_organization(resolver.resolve(org), meeting_id)
            for org in knowledge.entities.organizations
        ]

        # Execute all upserts concurrently (Memgraph handles parallel sessions)
        await asyncio.gather(*tasks, return_exceptions=True)

        # 8. Meeting similarity — fire separately (depends on Qdrant)
        asyncio.create_task(
            _link_similar(meeting_id, meta.get("summary", "")),
            name=f"graph_sim_{meeting_id[:8]}",
        )

        log.info("Graph sync complete meeting_id=%s", meeting_id)

    except Exception as exc:
        log.error("Graph sync failed meeting_id=%s: %s", meeting_id, exc)


async def _link_similar(meeting_id: str, summary: str, top_k: int = 5) -> None:
    if not summary:
        return
    try:
        from app.core.vector_store.embedder import MultiLevelEmbedder
        from app.core.vector_store.qdrant_store import QdrantMeetingStore
        from app.core.graph.client import get_graph_client

        embedder = MultiLevelEmbedder()
        vecs = await embedder.embed_quality([summary])
        store = QdrantMeetingStore()
        if not store.available:
            return

        hits = await asyncio.to_thread(
            store._client.search,
            collection_name="meetings_summaries",
            query_vector=vecs[0].tolist(),
            limit=top_k + 1,
        )
        client = get_graph_client()
        for hit in hits:
            other_id = hit.payload.get("meeting_id", "")
            if other_id and other_id != meeting_id and hit.score > 0.5:
                await client.link_similar(meeting_id, other_id, hit.score)
    except Exception as exc:
        log.debug("Meeting similarity link failed: %s", exc)
