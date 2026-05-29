"""
Cross-meeting intelligence analytics — PostgreSQL JSONB queries.

These functions mine the structured_data JSONB column on the meetings table,
making them available even when Memgraph is disabled.  They serve as the
fallback (and complement) to the graph layer.

All queries are async-safe: they use SQLAlchemy async sessions.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("intelligence.analytics")


async def get_speaker_analytics(speaker_name: str | None = None) -> list[dict]:
    """
    Speaker stats extracted from JSONB structured_data.

    Without a name → returns aggregated stats for every speaker across all meetings.
    With a name    → returns per-meeting breakdown for that speaker.
    """
    from app.db.database import AsyncSessionLocal
    from sqlalchemy import text

    async with AsyncSessionLocal() as session:
        if speaker_name:
            stmt = text(
                """
                SELECT m.id::text          AS meeting_id,
                       m.title             AS meeting_title,
                       m.date              AS meeting_date,
                       elem->>'name'       AS speaker,
                       (elem->>'speaking_percentage')::float  AS speaking_pct,
                       (elem->>'utterance_count')::int        AS utterances,
                       (elem->>'avg_sentiment')::float        AS sentiment,
                       (elem->>'word_count')::int             AS word_count
                FROM meetings m,
                     jsonb_array_elements(
                         m.structured_data->'speakers'
                     ) AS elem
                WHERE m.status = 'done'
                  AND m.structured_data IS NOT NULL
                  AND elem->>'name' ILIKE :pat
                ORDER BY m.date DESC
                LIMIT 100
                """
            )
            result = await session.execute(stmt, {"pat": f"%{speaker_name}%"})
        else:
            stmt = text(
                """
                SELECT elem->>'name'       AS speaker,
                       count(*)            AS meeting_count,
                       round(avg((elem->>'speaking_percentage')::float)::numeric, 2)
                                           AS avg_speaking_pct,
                       round(avg((elem->>'avg_sentiment')::float)::numeric, 3)
                                           AS avg_sentiment,
                       sum((elem->>'word_count')::int)
                                           AS total_words
                FROM meetings m,
                     jsonb_array_elements(
                         m.structured_data->'speakers'
                     ) AS elem
                WHERE m.status = 'done'
                  AND m.structured_data IS NOT NULL
                  AND elem->>'name' NOT LIKE 'SPEAKER_%'
                GROUP BY elem->>'name'
                ORDER BY meeting_count DESC
                LIMIT 50
                """
            )
            result = await session.execute(stmt)

        return [dict(row._mapping) for row in result.all()]


async def get_topic_analytics() -> list[dict]:
    """
    Topic frequency from JSONB structured_data.topics.

    Returns each unique topic title with meeting count and last occurrence.
    """
    from app.db.database import AsyncSessionLocal
    from sqlalchemy import text

    async with AsyncSessionLocal() as session:
        stmt = text(
            """
            SELECT elem->>'title'          AS topic,
                   count(*)               AS frequency,
                   max(m.date)            AS last_seen,
                   min(m.date)            AS first_seen,
                   array_agg(DISTINCT m.id::text) FILTER (WHERE m.id IS NOT NULL)
                                          AS meeting_ids
            FROM meetings m,
                 jsonb_array_elements(
                     m.structured_data->'topics'
                 ) AS elem
            WHERE m.status = 'done'
              AND m.structured_data IS NOT NULL
            GROUP BY elem->>'title'
            ORDER BY frequency DESC
            LIMIT 50
            """
        )
        result = await session.execute(stmt)
        return [dict(row._mapping) for row in result.all()]


async def get_meeting_trends_pg(days: int = 90) -> dict[str, Any]:
    """
    Weekly aggregates for meeting volume, duration, and sentiment.
    Pure PostgreSQL — no graph dependency.
    """
    from app.db.database import AsyncSessionLocal
    from sqlalchemy import text

    async with AsyncSessionLocal() as session:
        stmt = text(
            """
            SELECT date_trunc('week', date)         AS week,
                   count(*)                         AS meeting_count,
                   round(avg(duration_minutes)::numeric, 1)
                                                    AS avg_duration_min,
                   round(avg(
                       (structured_data->'sentiment'->>'overall_score')::float
                   )::numeric, 3)                   AS avg_sentiment
            FROM meetings
            WHERE status = 'done'
              AND date >= NOW() - ((:days)::int * INTERVAL '1 day')
            GROUP BY week
            ORDER BY week DESC
            """
        )
        result = await session.execute(stmt, {"days": days})
        weekly = [dict(row._mapping) for row in result.all()]

        # Aggregate entity frequency across JSONB for top organizations / tech
        stmt2 = text(
            """
            SELECT elem::text             AS org,
                   count(*)              AS frequency
            FROM meetings m,
                 jsonb_array_elements_text(
                     m.structured_data->'entities'->'organizations'
                 ) AS elem
            WHERE m.status = 'done'
              AND m.structured_data IS NOT NULL
              AND m.date >= NOW() - ((:days)::int * INTERVAL '1 day')
            GROUP BY elem
            ORDER BY frequency DESC
            LIMIT 20
            """
        )
        result2 = await session.execute(stmt2, {"days": days})
        top_orgs = [dict(row._mapping) for row in result2.all()]

        return {"weekly_trends": weekly, "top_organizations": top_orgs}


async def get_interaction_analysis(meeting_id: str) -> dict[str, Any]:
    """
    Speaker interaction patterns for a single meeting.
    Infers turn-taking from utterance sequence in structured_data.
    """
    from app.db.database import AsyncSessionLocal
    from sqlalchemy import text

    async with AsyncSessionLocal() as session:
        stmt = text(
            """
            SELECT structured_data->'speakers' AS speakers,
                   structured_data->'sentiment'->'by_speaker' AS sentiments
            FROM meetings
            WHERE id = :mid AND status = 'done'
            """
        )
        result = await session.execute(stmt, {"mid": meeting_id})
        row = result.one_or_none()
        if row is None:
            return {}

        speakers = row.speakers or []
        sentiments = row.sentiments or {}

        # Build speaker ranking by engagement
        ranked = sorted(
            [
                {
                    "name":        s.get("name"),
                    "speaking_pct": s.get("speaking_percentage", 0),
                    "utterances":  s.get("utterance_count", 0),
                    "words":       s.get("word_count", 0),
                    "sentiment":   sentiments.get(s.get("name", ""), 0.0),
                }
                for s in speakers
            ],
            key=lambda x: x["speaking_pct"],
            reverse=True,
        )
        dominant = ranked[0]["name"] if ranked else None
        return {
            "speaker_ranking": ranked,
            "dominant_speaker": dominant,
            "speaker_count":   len(ranked),
        }


async def get_knowledge_retention(topic: str, limit: int = 10) -> list[dict]:
    """
    What has been decided / actioned about a topic across all meetings?
    Mines JSONB for decisions and action items related to keyword.
    """
    from app.db.database import AsyncSessionLocal
    from sqlalchemy import text

    kw = f"%{topic.lower()}%"

    async with AsyncSessionLocal() as session:
        stmt = text(
            """
            SELECT m.id::text  AS meeting_id,
                   m.title     AS meeting_title,
                   m.date      AS meeting_date,
                   elem->>'text' AS item_text,
                   'decision'  AS item_type
            FROM meetings m,
                 jsonb_array_elements(m.structured_data->'candidate_decisions') AS elem
            WHERE m.status = 'done'
              AND lower(elem->>'text') LIKE :kw

            UNION ALL

            SELECT m.id::text  AS meeting_id,
                   m.title     AS meeting_title,
                   m.date      AS meeting_date,
                   elem->>'text' AS item_text,
                   'action_item' AS item_type
            FROM meetings m,
                 jsonb_array_elements(m.structured_data->'candidate_tasks') AS elem
            WHERE m.status = 'done'
              AND lower(elem->>'text') LIKE :kw

            ORDER BY meeting_date DESC
            LIMIT :limit
            """
        )
        result = await session.execute(stmt, {"kw": kw, "limit": limit})
        return [dict(row._mapping) for row in result.all()]
