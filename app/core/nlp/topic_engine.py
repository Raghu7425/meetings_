"""
Cross-meeting topic discovery via BERTopic.

Maintains a persistent BERTopic model that grows with each processed meeting.
Topics are discovered in batch and their evolution tracked over time.

Design for speed:
  - Model is NOT retrained on every meeting; it fits when corpus >= min_docs.
  - New meetings call transform() only (no refit) — fast inference path.
  - Refit is triggered in the background when corpus grows by REFIT_THRESHOLD.
  - Model is persisted to BERTOPIC_MODEL_DIR, reloads in < 2 s on restart.
  - Reuses the already-loaded all-MiniLM-L6-v2 embedder — no extra model cost.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("nlp.topic_engine")

REFIT_THRESHOLD = 10  # refit model after this many new documents since last fit

_engine: "TopicEngine | None" = None


def get_topic_engine() -> "TopicEngine":
    global _engine
    if _engine is None:
        from app.config import BERTOPIC_MODEL_DIR, BERTOPIC_MIN_TOPIC_SIZE
        _engine = TopicEngine(
            Path(BERTOPIC_MODEL_DIR),
            min_topic_size=BERTOPIC_MIN_TOPIC_SIZE,
        )
        _engine.load()
    return _engine


class TopicEngine:
    def __init__(self, model_dir: Path, min_topic_size: int = 3) -> None:
        self._model_dir = model_dir
        self._min_topic_size = min_topic_size
        self._model: Any = None
        self._fitted = False
        self._corpus: list[str] = []
        self._corpus_ids: list[str] = []
        self._docs_since_refit: int = 0

    # ── persistence ───────────────────────────────────────────────────────────

    def load(self) -> None:
        model_path = self._model_dir / "bertopic_model"
        corpus_path = self._model_dir / "corpus.json"

        if model_path.exists():
            try:
                from bertopic import BERTopic
                self._model = BERTopic.load(str(model_path))
                self._fitted = True
                log.info("BERTopic loaded from %s", model_path)
            except Exception as e:
                log.warning("BERTopic load failed: %s", e)

        if corpus_path.exists():
            try:
                data = json.loads(corpus_path.read_text(encoding="utf-8"))
                self._corpus = data.get("texts", [])
                self._corpus_ids = data.get("ids", [])
            except Exception:
                pass

    def save(self) -> None:
        self._model_dir.mkdir(parents=True, exist_ok=True)
        if self._model is not None and self._fitted:
            try:
                self._model.save(str(self._model_dir / "bertopic_model"))
            except Exception as e:
                log.warning("BERTopic save failed: %s", e)
        (self._model_dir / "corpus.json").write_text(
            json.dumps({"texts": self._corpus, "ids": self._corpus_ids}),
            encoding="utf-8",
        )

    # ── model lifecycle ───────────────────────────────────────────────────────

    def _build_model(self) -> Any:
        from bertopic import BERTopic
        from app.core.vector_store.embedder import _get_fast
        return BERTopic(
            embedding_model=_get_fast(),
            min_topic_size=self._min_topic_size,
            calculate_probabilities=False,
            verbose=False,
        )

    def _refit(self) -> None:
        if len(self._corpus) < self._min_topic_size * 2:
            return
        if self._model is None:
            self._model = self._build_model()
        try:
            self._model.fit_transform(self._corpus)
            self._fitted = True
            self._docs_since_refit = 0
            self.save()
            log.info("BERTopic refit on %d docs", len(self._corpus))
        except Exception as e:
            log.warning("BERTopic fit failed: %s", e)

    # ── public API ────────────────────────────────────────────────────────────

    def add_meeting(self, text: str, meeting_id: str) -> list[dict]:
        """Add meeting to corpus; trigger refit if threshold reached."""
        if meeting_id in self._corpus_ids:
            return self.get_meeting_topics(text)

        self._corpus.append(text)
        self._corpus_ids.append(meeting_id)
        self._docs_since_refit += 1

        # Refit when enough new data has accumulated
        if (not self._fitted) or (self._docs_since_refit >= REFIT_THRESHOLD):
            self._refit()

        return self.get_meeting_topics(text)

    def get_meeting_topics(self, text: str) -> list[dict]:
        """Fast transform for a single text — no refit."""
        if not self._fitted or self._model is None:
            return []
        try:
            topics, _ = self._model.transform([text])
            return self._topic_details(topics[0])
        except Exception:
            return []

    def get_all_topics(self) -> list[dict]:
        """Return all discovered topics with metadata."""
        if not self._fitted or self._model is None:
            return []
        try:
            info = self._model.get_topic_info()
            result = []
            for _, row in info.iterrows():
                if int(row["Topic"]) == -1:
                    continue
                words = self._model.get_topic(int(row["Topic"])) or []
                result.append({
                    "id": int(row["Topic"]),
                    "label": str(row.get("Name", f"Topic {row['Topic']}")),
                    "count": int(row["Count"]),
                    "keywords": [w for w, _ in words[:10]],
                })
            return result
        except Exception:
            return []

    def _topic_details(self, topic_id: int) -> list[dict]:
        if topic_id == -1 or self._model is None:
            return []
        words = self._model.get_topic(topic_id) or []
        return [{"id": topic_id, "keywords": [w for w, _ in words[:10]]}]

    # ── async wrappers (run in thread pool) ───────────────────────────────────

    async def add_meeting_async(self, text: str, meeting_id: str) -> list[dict]:
        return await asyncio.to_thread(self.add_meeting, text, meeting_id)

    async def get_all_topics_async(self) -> list[dict]:
        return await asyncio.to_thread(self.get_all_topics)
