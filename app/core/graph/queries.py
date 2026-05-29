"""
Knowledge graph query layer — answers intelligence questions via Cypher.

All queries return plain dicts (JSON-serializable).
When MEMGRAPH_ENABLED=false every function returns [] / {} immediately —
the intelligence endpoints degrade gracefully to PostgreSQL-only answers.
"""
from __future__ import annotations

import logging

log = logging.getLogger("graph.queries")


async def _run(query: str, params: dict | None = None) -> list[dict]:
    from app.core.graph.client import get_graph_client
    return await get_graph_client()._run(query, params)


# ── topic intelligence ────────────────────────────────────────────────────────

async def topic_history(topic: str) -> list[dict]:
    """Which meetings discussed this topic?"""
    return await _run(
        """
        MATCH (t:Topic)-[:DISCUSSED]-(m:Meeting)
        WHERE t.name = $topic OR t.name CONTAINS $topic
        RETURN m.id AS id, m.title AS title, m.date AS date, m.summary AS summary
        ORDER BY m.date DESC LIMIT 20
        """,
        {"topic": topic},
    )


async def topic_owners(topic: str) -> list[dict]:
    """Who owns action items that relate to meetings where this topic was discussed?"""
    return await _run(
        """
        MATCH (t:Topic)-[:DISCUSSED]-(m:Meeting)-[:HAS_ACTION]->(a:ActionItem)<-[:OWNS]-(p:Person)
        WHERE t.name = $topic OR t.name CONTAINS $topic
        RETURN p.canonical_name AS person, count(a) AS owned_count
        ORDER BY owned_count DESC LIMIT 10
        """,
        {"topic": topic},
    )


async def all_topics() -> list[dict]:
    """All topics with meeting-count and keywords."""
    return await _run(
        """
        MATCH (t:Topic)
        OPTIONAL MATCH (t)<-[:DISCUSSED]-(m:Meeting)
        RETURN t.name AS name, t.keywords AS keywords,
               count(m) AS meeting_count, t.first_seen AS first_seen
        ORDER BY meeting_count DESC
        """
    )


# ── speaker intelligence ──────────────────────────────────────────────────────

async def speaker_profile(name: str) -> dict:
    """Aggregate stats + top topics + action-item delivery rate."""
    stats = await _run(
        """
        MATCH (p:Person {canonical_name: $name})-[r:PARTICIPATED_IN]->(m:Meeting)
        RETURN count(m)             AS meeting_count,
               avg(r.speaking_pct) AS avg_speaking_pct,
               avg(r.sentiment)    AS avg_sentiment,
               sum(r.speaking_s)   AS total_speaking_s
        """,
        {"name": name},
    )
    top_topics = await _run(
        """
        MATCH (p:Person {canonical_name: $name})-[:PARTICIPATED_IN]->(m:Meeting)
              -[:DISCUSSED]->(t:Topic)
        RETURN t.name AS topic, count(m) AS frequency
        ORDER BY frequency DESC LIMIT 10
        """,
        {"name": name},
    )
    action_stats = await _run(
        """
        MATCH (p:Person {canonical_name: $name})-[:OWNS]->(a:ActionItem)
        RETURN count(a) AS total,
               sum(CASE WHEN a.status = 'done' THEN 1 ELSE 0 END) AS done_count
        """,
        {"name": name},
    )
    return {
        "name":         name,
        "stats":        stats[0] if stats else {},
        "top_topics":   top_topics,
        "action_items": action_stats[0] if action_stats else {},
    }


# ── meeting intelligence ──────────────────────────────────────────────────────

async def meeting_context(meeting_id: str) -> dict:
    """Full graph context for one meeting: participants, topics, decisions, risks, similar."""
    parts, topics, decisions, risks, similar = await _gather(
        _run(
            """
            MATCH (p:Person)-[r:PARTICIPATED_IN]->(m:Meeting {id: $id})
            RETURN p.canonical_name AS name, r.speaking_pct AS pct,
                   r.utterances AS utterances, r.sentiment AS sentiment
            ORDER BY r.speaking_pct DESC
            """,
            {"id": meeting_id},
        ),
        _run(
            """
            MATCH (m:Meeting {id: $id})-[:DISCUSSED]->(t:Topic)
            RETURN t.name AS name, t.keywords AS keywords
            """,
            {"id": meeting_id},
        ),
        _run(
            """
            MATCH (m:Meeting {id: $id})-[:RESULTED_IN]->(d:Decision)
            RETURN d.text AS text, d.confidence AS confidence
            """,
            {"id": meeting_id},
        ),
        _run(
            """
            MATCH (m:Meeting {id: $id})-[:RAISED]->(r:Risk)
            RETURN r.text AS text, r.category AS category, r.severity AS severity
            """,
            {"id": meeting_id},
        ),
        _run(
            """
            MATCH (m:Meeting {id: $id})-[r:SIMILAR_TO]-(o:Meeting)
            RETURN o.id AS id, o.title AS title, r.score AS score
            ORDER BY r.score DESC LIMIT 5
            """,
            {"id": meeting_id},
        ),
    )
    return {
        "participants":    parts,
        "topics":          topics,
        "decisions":       decisions,
        "risks":           risks,
        "similar_meetings": similar,
    }


# ── cross-meeting intelligence ────────────────────────────────────────────────

async def recurring_issues(min_frequency: int = 2) -> list[dict]:
    """Risks that appear in multiple meetings — persistent blockers."""
    return await _run(
        """
        MATCH (r:Risk)<-[:RAISED]-(m:Meeting)
        WITH r.category AS category, r.severity AS severity,
             collect(r.text)[0..3] AS examples, count(m) AS frequency
        WHERE frequency >= $min_freq
        RETURN category, severity, frequency, examples
        ORDER BY frequency DESC LIMIT 20
        """,
        {"min_freq": min_frequency},
    )


async def historical_decisions(limit: int = 50) -> list[dict]:
    """All decisions ever made, newest first."""
    return await _run(
        """
        MATCH (d:Decision)<-[:RESULTED_IN]-(m:Meeting)
        RETURN d.text AS decision, d.confidence AS confidence,
               m.title AS meeting, m.date AS date
        ORDER BY m.date DESC LIMIT $limit
        """,
        {"limit": limit},
    )


async def meeting_trends(days: int = 90) -> list[dict]:
    """Raw meeting nodes for trend analysis — sorted by date."""
    return await _run(
        """
        MATCH (m:Meeting)
        RETURN m.date AS date, m.duration_min AS duration_min,
               m.sentiment_label AS sentiment, m.engagement AS engagement
        ORDER BY m.date DESC LIMIT 200
        """
    )


# ── helper ────────────────────────────────────────────────────────────────────

async def _gather(*coros):
    import asyncio
    results = await asyncio.gather(*coros, return_exceptions=True)
    return [r if not isinstance(r, Exception) else [] for r in results]
