"""
Multi-level embedding service.

Produces three embedding levels per meeting — mirroring the three Qdrant
collections:

  Level       Model                    Use-case
  ─────────   ──────────────────────   ─────────────────────────────────────
  fast        all-MiniLM-L6-v2  384d  Real-time Q&A, voice agent retrieval
  quality     all-mpnet-base-v2 768d  Semantic + summary collections
  summary     all-mpnet-base-v2 768d  Summary-level retrieval

Models are loaded lazily and cached for the process lifetime.
CPU-heavy encoding is always run in a thread so it never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBED_MODEL_FAST,
    EMBED_MODEL_QUALITY,
    EMBED_DIM_SMALL,
    EMBED_DIM_LARGE,
)

log = logging.getLogger("embedder")

_fast_model:    SentenceTransformer | None = None
_quality_model: SentenceTransformer | None = None


def _get_fast() -> SentenceTransformer:
    global _fast_model
    if _fast_model is None:
        log.info("[embedder] Loading fast model: %s", EMBED_MODEL_FAST)
        _fast_model = SentenceTransformer(EMBED_MODEL_FAST, device="cpu")
    return _fast_model


def _get_quality() -> SentenceTransformer:
    global _quality_model
    if _quality_model is None:
        log.info("[embedder] Loading quality model: %s", EMBED_MODEL_QUALITY)
        _quality_model = SentenceTransformer(EMBED_MODEL_QUALITY, device="cpu")
    return _quality_model


# ── sync helpers (run in threads) ─────────────────────────────────────────────

def _encode_fast_sync(texts: list[str]) -> np.ndarray:
    model = _get_fast()
    return model.encode(texts, normalize_embeddings=True, batch_size=64).astype("float32")


def _encode_quality_sync(texts: list[str]) -> np.ndarray:
    model = _get_quality()
    return model.encode(texts, normalize_embeddings=True, batch_size=32).astype("float32")


# ── public async API ──────────────────────────────────────────────────────────

class MultiLevelEmbedder:
    """
    Async wrapper around both embedding models.

    Example:
        embedder = MultiLevelEmbedder()
        fast_vecs = await embedder.embed_fast(["hello world"])
        qual_vecs = await embedder.embed_quality(["full transcript chunk…"])
    """

    async def embed_fast(self, texts: Sequence[str]) -> np.ndarray:
        """384-d embeddings via all-MiniLM-L6-v2."""
        if not texts:
            return np.empty((0, EMBED_DIM_SMALL), dtype="float32")
        return await asyncio.to_thread(_encode_fast_sync, list(texts))

    async def embed_quality(self, texts: Sequence[str]) -> np.ndarray:
        """768-d embeddings via all-mpnet-base-v2."""
        if not texts:
            return np.empty((0, EMBED_DIM_LARGE), dtype="float32")
        return await asyncio.to_thread(_encode_quality_sync, list(texts))

    async def embed_for_query(self, query: str) -> np.ndarray:
        """Fast 384-d single-query embedding (used in retrieval hot-path)."""
        return await self.embed_fast([query])

    async def embed_summary(self, summary_text: str) -> np.ndarray:
        """Quality 768-d single-document embedding for summary collection."""
        return await self.embed_quality([summary_text])

    @staticmethod
    def preload() -> None:
        """Eagerly load both models (call at app startup)."""
        _get_fast()
        _get_quality()
