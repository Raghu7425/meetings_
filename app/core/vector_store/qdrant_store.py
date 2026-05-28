"""
Qdrant vector store — three-collection hierarchical storage.

Collections
───────────
  meetings_raw       — verbatim transcript windows (fast 384-d model)
  meetings_semantic  — cleaned semantic chunks (quality 768-d model)
  meetings_summaries — summary/structured sections (quality 768-d model)

Each point carries a rich payload for metadata filtering:
  job_id, filename, speaker, start_sec, end_sec, chunk_type, section, chunk_index

Retrieval
─────────
  search_hybrid() runs a combined search across all three collections,
  deduplicates by text hash, and returns results ranked by:
    1. Collection priority (structured > semantic > raw)
    2. Cosine similarity score

The FAISS fallback in agent.py is preserved for environments where Qdrant
is not available.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any, Sequence

from app.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_API_KEY,
    QDRANT_PREFER_GRPC,
    QDRANT_COL_RAW,
    QDRANT_COL_SEMANTIC,
    QDRANT_COL_SUMMARY,
    EMBED_DIM_SMALL,
    EMBED_DIM_LARGE,
    RETRIEVAL_TOP_K,
)
from app.core.vector_store.chunker import SemanticChunker, TextChunk, ChunkType
from app.core.vector_store.embedder import MultiLevelEmbedder

log = logging.getLogger("qdrant_store")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
    )
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False
    log.warning("[qdrant] qdrant-client not installed — vector store disabled")


def _point_id(job_id: str, chunk_index: int, collection: str) -> str:
    """Stable deterministic UUID-like ID for a point."""
    raw = f"{collection}:{job_id}:{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _chunk_to_payload(chunk: TextChunk) -> dict[str, Any]:
    return {
        "text":        chunk.text,
        "job_id":      chunk.job_id,
        "filename":    chunk.filename,
        "speaker":     chunk.speaker,
        "start_sec":   chunk.start_sec,
        "end_sec":     chunk.end_sec,
        "chunk_type":  chunk.chunk_type.value,
        "section":     chunk.section,
        "chunk_index": chunk.chunk_index,
    }


class QdrantMeetingStore:
    """
    Async facade over QdrantClient.

    All heavy encode+upsert work is awaited correctly.
    Gracefully degrades to no-op if Qdrant is unavailable.
    """

    def __init__(self) -> None:
        self._client: Any | None = None
        if _QDRANT_AVAILABLE:
            try:
                self._client = QdrantClient(
                    host=QDRANT_HOST,
                    port=QDRANT_PORT,
                    api_key=QDRANT_API_KEY or None,
                    prefer_grpc=QDRANT_PREFER_GRPC,
                    timeout=30,
                )
                log.info("[qdrant] client initialised → %s:%d", QDRANT_HOST, QDRANT_PORT)
            except Exception as exc:
                log.warning("[qdrant] client init failed: %s — store disabled", exc)
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    # ── Startup ────────────────────────────────────────────────────────────────

    def ensure_collections(self) -> None:
        """Create collections if they don't exist (call at startup)."""
        if not self.available:
            return
        specs = [
            (QDRANT_COL_RAW,      EMBED_DIM_SMALL),
            (QDRANT_COL_SEMANTIC, EMBED_DIM_LARGE),
            (QDRANT_COL_SUMMARY,  EMBED_DIM_LARGE),
        ]
        existing = {c.name for c in self._client.get_collections().collections}
        for name, dim in specs:
            if name not in existing:
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
                log.info("[qdrant] created collection %s (dim=%d)", name, dim)

    # ── Index meeting ──────────────────────────────────────────────────────────

    async def index_meeting(
        self,
        job_id: str,
        filename: str,
        transcript: str,
        segments: list[dict],
        report: Any,  # MeetingReport
        embedder: MultiLevelEmbedder,
        chunker: SemanticChunker,
    ) -> None:
        if not self.available:
            log.debug("[qdrant] skipping index — not available")
            return

        raw_chunks, semantic_chunks = chunker.chunk_transcript(
            transcript, job_id=job_id, filename=filename, segments=segments
        )
        structured_chunks = chunker.chunk_report(report, job_id=job_id, filename=filename)

        # Parallel encode all three levels
        raw_texts        = [c.text for c in raw_chunks]
        semantic_texts   = [c.text for c in semantic_chunks]
        structured_texts = [c.text for c in structured_chunks]

        raw_vecs, sem_vecs, struct_vecs = await asyncio.gather(
            embedder.embed_fast(raw_texts),
            embedder.embed_quality(semantic_texts),
            embedder.embed_quality(structured_texts),
        )

        # Upsert all three collections concurrently
        await asyncio.gather(
            asyncio.to_thread(self._upsert, QDRANT_COL_RAW, raw_chunks, raw_vecs),
            asyncio.to_thread(self._upsert, QDRANT_COL_SEMANTIC, semantic_chunks, sem_vecs),
            asyncio.to_thread(self._upsert, QDRANT_COL_SUMMARY, structured_chunks, struct_vecs),
        )

        log.info(
            "[qdrant] indexed job=%s raw=%d semantic=%d structured=%d",
            job_id, len(raw_chunks), len(semantic_chunks), len(structured_chunks),
        )

    # ── Retrieval ──────────────────────────────────────────────────────────────

    async def search_hybrid(
        self,
        query: str,
        job_id: str,
        embedder: MultiLevelEmbedder,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Query all three collections, deduplicate, and return ranked results.
        Structured chunks are boosted (they are more directly informative).
        """
        if not self.available:
            return []

        fast_vec = (await embedder.embed_fast([query]))[0].tolist()
        qual_vec = (await embedder.embed_quality([query]))[0].tolist()

        job_filter = Filter(
            must=[FieldCondition(key="job_id", match=MatchValue(value=job_id))]
        )

        raw_hits, sem_hits, struct_hits = await asyncio.gather(
            asyncio.to_thread(self._search, QDRANT_COL_RAW,      fast_vec, top_k, job_filter),
            asyncio.to_thread(self._search, QDRANT_COL_SEMANTIC,  qual_vec, top_k, job_filter),
            asyncio.to_thread(self._search, QDRANT_COL_SUMMARY,   qual_vec, top_k, job_filter),
        )

        # Deduplicate by text hash; structured > semantic > raw priority
        seen:    set[str] = set()
        results: list[dict] = []

        for hits, boost in [(struct_hits, 0.2), (sem_hits, 0.1), (raw_hits, 0.0)]:
            for hit in hits:
                text = hit.payload.get("text", "")
                key  = hashlib.md5(text.encode()).hexdigest()
                if key not in seen:
                    seen.add(key)
                    results.append({
                        "text":    text,
                        "score":   hit.score + boost,
                        "section": hit.payload.get("section", ""),
                        "speaker": hit.payload.get("speaker", ""),
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def delete_meeting(self, job_id: str) -> None:
        """Remove all vectors for a job from all collections."""
        if not self.available:
            return
        job_filter = Filter(
            must=[FieldCondition(key="job_id", match=MatchValue(value=job_id))]
        )
        for col in (QDRANT_COL_RAW, QDRANT_COL_SEMANTIC, QDRANT_COL_SUMMARY):
            await asyncio.to_thread(self._client.delete, col, points_selector=job_filter)
        log.info("[qdrant] deleted vectors job=%s", job_id)

    # ── sync helpers (run in thread) ───────────────────────────────────────────

    def _upsert(self, collection: str, chunks: list[TextChunk], vectors) -> None:
        import numpy as np
        points = [
            PointStruct(
                id=_point_id(c.job_id, c.chunk_index, collection),
                vector=vectors[i].tolist(),
                payload=_chunk_to_payload(c),
            )
            for i, c in enumerate(chunks)
        ]
        if not points:
            return
        # Batch in groups of 100 to avoid large payloads
        for batch_start in range(0, len(points), 100):
            self._client.upsert(
                collection_name=collection,
                points=points[batch_start : batch_start + 100],
            )

    def _search(self, collection: str, vector: list[float], top_k: int, filter_: Any):
        return self._client.search(
            collection_name=collection,
            query_vector=vector,
            query_filter=filter_,
            limit=top_k,
            with_payload=True,
        )


# ── singleton ──────────────────────────────────────────────────────────────────

_store: QdrantMeetingStore | None = None


def get_qdrant_store() -> QdrantMeetingStore:
    """Return the process-level singleton (lazy init)."""
    global _store
    if _store is None:
        _store = QdrantMeetingStore()
    return _store
