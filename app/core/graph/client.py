"""
Memgraph client — async Cypher via Bolt protocol (neo4j driver).

Memgraph speaks the Bolt protocol, so the official neo4j Python async driver
works without modification.  All write operations use MERGE so they are safe
to retry (idempotent upserts).

Schema is bootstrapped on first connection:
  Nodes:  Meeting  Person  Topic  Organization  Decision  ActionItem  Risk
  Edges:  PARTICIPATED_IN  DISCUSSED  RESULTED_IN  HAS_ACTION
          OWNS  RAISED  SIMILAR_TO  MENTIONED_IN  EVOLVED_INTO
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger("graph.client")

_driver: Any = None
_driver_lock = asyncio.Lock()

# ── driver singleton ──────────────────────────────────────────────────────────


async def get_graph_driver():
    global _driver
    if _driver is not None:
        return _driver
    async with _driver_lock:
        if _driver is not None:
            return _driver
        from neo4j import AsyncGraphDatabase
        from app.config import (
            MEMGRAPH_HOST,
            MEMGRAPH_PORT,
            MEMGRAPH_USER,
            MEMGRAPH_PASSWORD,
        )
        uri = f"bolt://{MEMGRAPH_HOST}:{MEMGRAPH_PORT}"
        auth = (MEMGRAPH_USER, MEMGRAPH_PASSWORD) if MEMGRAPH_USER else ("", "")
        _driver = AsyncGraphDatabase.driver(uri, auth=auth)
        await _bootstrap_schema(_driver)
        log.info("Memgraph driver connected: %s", uri)
    return _driver


async def close_graph_driver() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


async def _bootstrap_schema(driver) -> None:
    """Create indexes and uniqueness constraints once on first connect."""
    ddl = [
        # Memgraph uses CREATE CONSTRAINT syntax (not ASSERT … IS UNIQUE in v2.x)
        "CREATE INDEX ON :Meeting(id)",
        "CREATE INDEX ON :Meeting(date)",
        "CREATE INDEX ON :Person(canonical_name)",
        "CREATE INDEX ON :Topic(name)",
        "CREATE INDEX ON :Organization(canonical_name)",
        "CREATE INDEX ON :ActionItem(status)",
        "CREATE INDEX ON :Risk(category)",
    ]
    async with driver.session() as session:
        for stmt in ddl:
            try:
                await session.run(stmt)
            except Exception:
                pass  # index may already exist


# ── client class ──────────────────────────────────────────────────────────────


class GraphClient:
    """Async Cypher client for meeting knowledge graph operations."""

    async def _run(self, query: str, params: dict | None = None) -> list[dict]:
        from app.config import MEMGRAPH_ENABLED
        if not MEMGRAPH_ENABLED:
            return []
        try:
            driver = await get_graph_driver()
            async with driver.session() as session:
                result = await session.run(query, params or {})
                return await result.data()
        except Exception as exc:
            log.debug("Graph query failed: %s | %s", exc, query[:120])
            return []

    # ── node upserts ──────────────────────────────────────────────────────────

    async def upsert_meeting(self, d: dict) -> None:
        await self._run(
            """
            MERGE (m:Meeting {id: $id})
            SET m.call_id         = $call_id,
                m.title           = $title,
                m.date            = $date,
                m.duration_min    = $duration_min,
                m.summary         = $summary,
                m.sentiment_label = $sentiment_label,
                m.engagement      = $engagement,
                m.platform        = $platform
            """,
            d,
        )

    async def upsert_person(self, canonical_name: str, email: str = "") -> None:
        await self._run(
            """
            MERGE (p:Person {canonical_name: $canonical_name})
            SET p.email = $email
            """,
            {"canonical_name": canonical_name, "email": email},
        )

    async def upsert_topic(
        self, name: str, keywords: list[str], meeting_id: str, duration_s: float
    ) -> None:
        await self._run(
            """
            MERGE (t:Topic {name: $name})
            ON CREATE SET t.first_seen = $now, t.keywords = $keywords, t.frequency = 1
            ON MATCH  SET t.last_seen  = $now,
                          t.frequency  = coalesce(t.frequency, 0) + 1
            """,
            {"name": name, "keywords": keywords, "now": _now()},
        )
        await self._run(
            """
            MATCH (m:Meeting {id: $meeting_id})
            MERGE (t:Topic {name: $name})
            MERGE (m)-[r:DISCUSSED]->(t)
            SET r.duration_s = $duration_s
            """,
            {"meeting_id": meeting_id, "name": name, "duration_s": duration_s},
        )

    async def upsert_organization(self, name: str) -> None:
        await self._run(
            "MERGE (o:Organization {canonical_name: $name})",
            {"name": name},
        )

    # ── edge creation ─────────────────────────────────────────────────────────

    async def link_participant(
        self, meeting_id: str, canonical_name: str, stats: dict
    ) -> None:
        await self._run(
            """
            MERGE (p:Person {canonical_name: $name})
            WITH p
            MATCH (m:Meeting {id: $meeting_id})
            MERGE (p)-[r:PARTICIPATED_IN]->(m)
            SET r.speaking_s   = $speaking_s,
                r.speaking_pct = $speaking_pct,
                r.utterances   = $utterances,
                r.sentiment    = $sentiment
            """,
            {
                "name":        canonical_name,
                "meeting_id":  meeting_id,
                "speaking_s":  stats.get("speaking_time", 0.0),
                "speaking_pct": stats.get("speaking_pct", 0.0),
                "utterances":  stats.get("utterance_count", 0),
                "sentiment":   stats.get("sentiment_score", 0.0),
            },
        )

    async def add_decision(
        self,
        meeting_id: str,
        text: str,
        confidence: float,
        speaker: str | None,
    ) -> None:
        await self._run(
            """
            MATCH (m:Meeting {id: $meeting_id})
            CREATE (d:Decision {text: $text, confidence: $conf, date: $now})
            CREATE (m)-[:RESULTED_IN]->(d)
            """,
            {"meeting_id": meeting_id, "text": text[:500], "conf": confidence, "now": _now()},
        )
        if speaker:
            await self._run(
                """
                MATCH (m:Meeting {id: $mid})-[:RESULTED_IN]->(d:Decision {text: $text})
                MERGE (p:Person {canonical_name: $speaker})
                MERGE (d)-[:RAISED_BY]->(p)
                """,
                {"mid": meeting_id, "text": text[:500], "speaker": speaker},
            )

    async def add_action_item(
        self,
        meeting_id: str,
        task: str,
        owner: str | None,
        priority: str,
        deadline: str | None,
    ) -> None:
        await self._run(
            """
            MATCH (m:Meeting {id: $meeting_id})
            CREATE (a:ActionItem {task: $task, priority: $priority,
                                  status: 'open', deadline: $deadline})
            CREATE (m)-[:HAS_ACTION]->(a)
            """,
            {
                "meeting_id": meeting_id,
                "task":       task[:500],
                "priority":   priority,
                "deadline":   deadline or "",
            },
        )
        if owner:
            await self._run(
                """
                MATCH (m:Meeting {id: $mid})-[:HAS_ACTION]->(a:ActionItem {task: $task})
                MERGE (p:Person {canonical_name: $owner})
                MERGE (p)-[:OWNS]->(a)
                """,
                {"mid": meeting_id, "task": task[:500], "owner": owner},
            )

    async def add_risk(
        self,
        meeting_id: str,
        text: str,
        category: str,
        severity: str,
    ) -> None:
        await self._run(
            """
            MATCH (m:Meeting {id: $meeting_id})
            CREATE (r:Risk {text: $text, category: $category, severity: $severity})
            CREATE (m)-[:RAISED]->(r)
            """,
            {"meeting_id": meeting_id, "text": text[:400], "category": category, "severity": severity},
        )

    async def link_organization(self, org_name: str, meeting_id: str) -> None:
        await self._run(
            """
            MERGE (o:Organization {canonical_name: $name})
            WITH o
            MATCH (m:Meeting {id: $meeting_id})
            MERGE (o)-[:MENTIONED_IN]->(m)
            """,
            {"name": org_name, "meeting_id": meeting_id},
        )

    async def link_similar(
        self, meeting_id: str, other_id: str, score: float
    ) -> None:
        if meeting_id == other_id:
            return
        await self._run(
            """
            MATCH (m1:Meeting {id: $m1}), (m2:Meeting {id: $m2})
            MERGE (m1)-[r:SIMILAR_TO]-(m2)
            SET r.score = $score
            """,
            {"m1": meeting_id, "m2": other_id, "score": round(score, 4)},
        )


# ── module-level singleton ────────────────────────────────────────────────────

_client: GraphClient | None = None


def get_graph_client() -> GraphClient:
    global _client
    if _client is None:
        _client = GraphClient()
    return _client


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
