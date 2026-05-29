"""
Entity normalization via SentenceTransformers cosine similarity.

Resolves STT-induced surface variations to a single canonical form:
  "Pete" / "Peter" / "Pita"     → "Peter Smith"  (if sim > threshold)
  "Google" / "Google LLC" / "Alphabet" kept separate (sim < threshold)

Uses the already-loaded all-MiniLM-L6-v2 model (reused from embedder —
no extra model load).  Results are cached in-process; the canonical map
can be serialized to Redis for cross-restart persistence.

Thread-safe: all mutations are protected by a threading.Lock.
"""
from __future__ import annotations

import logging
from threading import Lock

import numpy as np

log = logging.getLogger("nlp.entity_resolver")

_resolver: "EntityResolver | None" = None
_resolver_lock = Lock()


def get_entity_resolver() -> "EntityResolver":
    global _resolver
    if _resolver is None:
        with _resolver_lock:
            if _resolver is None:
                from app.core.vector_store.embedder import _get_fast
                from app.config import ENTITY_RESOLVER_THRESHOLD
                _resolver = EntityResolver(_get_fast(), ENTITY_RESOLVER_THRESHOLD)
    return _resolver


class EntityResolver:
    """
    Deduplicates entity surface forms using embedding cosine similarity.

    canonical_map : raw_mention  → canonical_name
    _embeddings   : canonical_name → unit-norm embedding (float32)
    """

    def __init__(self, model, threshold: float = 0.82) -> None:
        self._model = model
        self._threshold = threshold
        self._lock = Lock()
        self._canonical_map: dict[str, str] = {}
        self._embeddings: dict[str, np.ndarray] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def resolve(self, name: str) -> str:
        name = name.strip()
        if not name:
            return name

        with self._lock:
            if name in self._canonical_map:
                return self._canonical_map[name]

            emb = self._encode(name)
            best_sim, best_canonical = self._nearest(emb)

            if best_canonical is not None and best_sim >= self._threshold:
                # Prefer the longer (more complete) surface form as canonical
                if len(name) > len(best_canonical):
                    self._promote_canonical(best_canonical, name, emb)
                    return name
                self._canonical_map[name] = best_canonical
                return best_canonical

            # First occurrence — register as new canonical
            self._canonical_map[name] = name
            self._embeddings[name] = emb
            return name

    def resolve_list(self, names: list[str]) -> list[str]:
        return [self.resolve(n) for n in names]

    def force_alias(self, alias: str, canonical: str) -> None:
        """Manually pin an alias to a canonical name."""
        with self._lock:
            self._canonical_map[alias.strip()] = canonical.strip()

    def snapshot(self) -> dict[str, str]:
        """Return a copy of the canonical map (for Redis persistence)."""
        with self._lock:
            return dict(self._canonical_map)

    def load_snapshot(self, snapshot: dict[str, str]) -> None:
        """Restore a previously saved canonical map."""
        with self._lock:
            self._canonical_map.update(snapshot)

    # ── internals ─────────────────────────────────────────────────────────────

    def _encode(self, text: str) -> np.ndarray:
        return self._model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

    def _nearest(self, emb: np.ndarray) -> tuple[float, str | None]:
        best_sim, best = 0.0, None
        for canonical, c_emb in self._embeddings.items():
            sim = float(np.dot(emb, c_emb))  # both unit-norm → cosine
            if sim > best_sim:
                best_sim, best = sim, canonical
        return best_sim, best

    def _promote_canonical(
        self, old: str, new: str, new_emb: np.ndarray
    ) -> None:
        """Replace `old` canonical with the longer `new` form."""
        self._embeddings.pop(old, None)
        self._embeddings[new] = new_emb
        for k in list(self._canonical_map):
            if self._canonical_map[k] == old:
                self._canonical_map[k] = new
        self._canonical_map[new] = new
