"""
Meeting business-logic service layer.

Encapsulates:
  - Job lifecycle management (create, read, delete via RedisJobStore)
  - RAG query routing (Qdrant → FAISS → transcript fallback)
  - Knowledge-base file management
  - Qdrant cleanup on job deletion

All methods are async and dependency-injection ready.  The upload API and
future CLI / test harnesses import from here rather than duplicating logic.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from app.config import MEETINGS_INPUT_DIR, UPLOADS_DIR
from app.core.pipeline.event_bus import PipelineEvent
from app.db.redis_client import RedisJobStore, get_redis

log = logging.getLogger("meeting_service")


class MeetingService:
    """Stateless service — instantiate per-request or use the module-level singleton."""

    # ── Job state ──────────────────────────────────────────────────────────────

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        r = await get_redis()
        store = RedisJobStore(r)
        return await store.get(job_id)

    async def is_done(self, job_id: str) -> bool:
        data = await self.get_job(job_id)
        if not data:
            return False
        return data.get("status") == PipelineEvent.DONE.value

    async def get_structured_data(self, job_id: str) -> dict | None:
        data = await self.get_job(job_id)
        if not data:
            return None
        raw = data.get("structured_data", "")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def delete_job(self, job_id: str) -> bool:
        r = await get_redis()
        store = RedisJobStore(r)
        data = await store.get(job_id)
        if not data:
            return False

        await store.delete(job_id)
        self._remove_files(job_id, data)
        await self._delete_vectors(job_id)
        self._invalidate_agent_index()
        return True

    # ── RAG query ──────────────────────────────────────────────────────────────

    async def retrieve_context(
        self,
        job_id: str,
        question: str,
        top_k: int = 10,
    ) -> str:
        """
        Return the best-matching context string for Q&A.
        Tries Qdrant first, then FAISS legacy, then raw transcript.
        """
        parts: list[str] = []
        data = await self.get_job(job_id)
        summary_txt = (data or {}).get("summary", "").strip()

        # ── Qdrant hybrid search ──
        try:
            from app.core.vector_store.qdrant_store import get_qdrant_store
            from app.core.vector_store.embedder import MultiLevelEmbedder

            qdrant = get_qdrant_store()
            if qdrant.available:
                embedder = MultiLevelEmbedder()
                hits = await qdrant.search_hybrid(question, job_id, embedder, top_k=top_k)
                if hits:
                    if summary_txt:
                        parts.append(f"MEETING SUMMARY:\n{summary_txt}")
                    parts.append(
                        "RELEVANT SECTIONS:\n\n" +
                        "\n\n".join(
                            f"[{h.get('section','').upper()}] {h['text']}"
                            for h in hits
                        )
                    )
                    return "\n\n---\n\n".join(parts)
        except Exception as exc:
            log.warning("qdrant retrieval failed job=%s: %s", job_id, exc)

        # ── FAISS fallback ──
        try:
            from app.api._legacy_rag import get_job_rag_index
            from app.core.vector_store.embedder import MultiLevelEmbedder
            import numpy as np, faiss

            idx, chunks = get_job_rag_index(job_id)
            if idx and chunks:
                embedder = MultiLevelEmbedder()
                q_emb = np.array((await embedder.embed_fast([question])).tolist(), dtype="float32")
                _, indices = idx.search(q_emb, min(top_k, len(chunks)))
                rag_ctx = "\n\n".join(chunks[i] for i in indices[0] if i < len(chunks))
                if summary_txt:
                    parts.append(f"MEETING SUMMARY:\n{summary_txt}")
                if rag_ctx:
                    parts.append(f"RELEVANT SECTIONS:\n{rag_ctx}")
                if parts:
                    return "\n\n---\n\n".join(parts)
        except Exception as exc:
            log.warning("faiss fallback failed job=%s: %s", job_id, exc)

        # ── Raw transcript fallback ──
        if summary_txt:
            parts.append(f"MEETING SUMMARY:\n{summary_txt}")
        if data:
            path = data.get("transcript_path", "")
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    parts.append(f"TRANSCRIPT:\n{f.read(6000)}")

        return "\n\n---\n\n".join(parts) if parts else "No meeting content available."

    # ── helpers ────────────────────────────────────────────────────────────────

    def _remove_files(self, job_id: str, data: dict) -> None:
        for path in [
            data.get("transcript_path", ""),
            os.path.join(MEETINGS_INPUT_DIR, f"{job_id}.txt"),
            os.path.join(UPLOADS_DIR, f"{job_id}.txt"),
        ]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError as exc:
                    log.warning("could not remove file %s: %s", path, exc)

    async def _delete_vectors(self, job_id: str) -> None:
        try:
            from app.core.vector_store.qdrant_store import get_qdrant_store
            qdrant = get_qdrant_store()
            if qdrant.available:
                await qdrant.delete_meeting(job_id)
        except Exception as exc:
            log.warning("vector delete failed job=%s: %s", job_id, exc)

    def _invalidate_agent_index(self) -> None:
        try:
            from app.core.agent import invalidate_agent_index
            invalidate_agent_index()
        except Exception:
            pass


# ── module-level singleton ─────────────────────────────────────────────────────
meeting_service = MeetingService()
