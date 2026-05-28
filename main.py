"""
FastAPI application entry point.

Startup sequence:
  1. Configure structured JSON logging
  2. Connect Redis (required for job state + pipeline events)
  3. TTS health probe (non-fatal)
  4. Preload STT / Agent / Speaker models (non-fatal)
  5. Ensure Qdrant collections exist (non-fatal)
  6. Run Alembic migrations (non-fatal)
  7. Initialise PostgreSQL tables (non-fatal)
  8. Ensure MinIO bucket exists (non-fatal)
  9. Start APScheduler (non-fatal)

Shutdown:
  - APScheduler stop
  - Redis connection pool close
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import logging
from contextlib import asynccontextmanager
from warnings import filterwarnings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

filterwarnings("ignore")

# ── Structured logging (must be first) ────────────────────────────────────────
from app.utils.logging_config import configure_logging
configure_logging()

from app.config import (
    APP_TITLE,
    APP_VERSION,
    STATIC_DIR,
)
from app.api.routes import router as rest_router
from app.api.upload import router as upload_router
from app.api.ws_progress import router as ws_progress_router
from app.db.redis_client import close_redis, get_redis, ping_redis

log = logging.getLogger("main")


# ── Optional feature guards ────────────────────────────────────────────────────

try:
    from app.api.websocket import router as ws_router
    from app.core.tts import synthesize_chunks, get_engine_name, get_edge_voice
    from app.core.stt import get_model
    from app.core.agent import ensure_system_initialized
    from app.core.speaker import get_classifier
    _VOICE_AVAILABLE = True
except ImportError as _e:
    ws_router = None
    _VOICE_AVAILABLE = False
    logging.getLogger("main").warning(
        "Voice assistant disabled (missing packages: %s). "
        "Video upload + transcription fully available.", _e,
    )

try:
    from app.api.webhook import router as webhook_router
    from app.api.meetings import router as meetings_router, action_router
    from app.core.scheduler import start_scheduler, stop_scheduler
    from app.core.storage import ensure_bucket
    from app.db.database import init_db
    _MEETING_PIPELINE_AVAILABLE = True
except ImportError:
    webhook_router = meetings_router = action_router = None
    _MEETING_PIPELINE_AVAILABLE = False


# ── Startup helpers ────────────────────────────────────────────────────────────

def _run_alembic_upgrade() -> None:
    from alembic.config import Config
    from alembic import command
    cfg = Config(os.path.join(os.path.dirname(__file__), "alembic.ini"))
    cfg.set_main_option(
        "script_location",
        os.path.join(os.path.dirname(__file__), "alembic"),
    )
    command.upgrade(cfg, "head")


async def _probe_tts() -> None:
    if not _VOICE_AVAILABLE:
        return
    log.info("TTS startup probe voice=%r", get_edge_voice())
    try:
        data = await synthesize_chunks("Hello.")
        log.info("TTS OK engine=%s bytes=%d", get_engine_name(), len(data) if data else 0)
    except Exception as exc:
        log.warning("TTS probe failed (non-fatal): %s", exc)


async def _preload_models() -> None:
    if not _VOICE_AVAILABLE:
        log.info("slim mode — skipping voice model preload")
        return
    log.info("preloading STT / Agent / Speaker models…")
    stt, agent, speaker = await asyncio.gather(
        asyncio.to_thread(get_model),
        ensure_system_initialized(),
        asyncio.to_thread(get_classifier),
        return_exceptions=True,
    )
    if isinstance(stt, Exception):
        raise RuntimeError(f"STT preload failed: {stt}")
    log.info("STT ready")
    if isinstance(agent, Exception):
        log.warning("Ollama unavailable — voice disabled: %s", agent)
    else:
        log.info("Agent/RAG/Ollama ready")
    if isinstance(speaker, Exception):
        log.warning("Speaker model not loaded (non-fatal): %s", speaker)
    else:
        log.info("Speaker model ready")


async def _ensure_qdrant_collections() -> None:
    try:
        from app.core.vector_store.qdrant_store import QdrantMeetingStore
        store = QdrantMeetingStore()
        if store.available:
            await asyncio.to_thread(store.ensure_collections)
            log.info("Qdrant collections ensured")
        else:
            log.warning("Qdrant not reachable — vector store disabled")
    except Exception as exc:
        log.warning("Qdrant setup failed (non-fatal): %s", exc)


async def _preload_embedders() -> None:
    try:
        from app.core.vector_store.embedder import MultiLevelEmbedder
        await asyncio.to_thread(MultiLevelEmbedder.preload)
        log.info("embedding models preloaded")
    except Exception as exc:
        log.warning("embedder preload failed (non-fatal): %s", exc)


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== %s v%s starting up ===", APP_TITLE, APP_VERSION)

    # Redis — required
    if not await ping_redis():
        log.warning("Redis not reachable — job state and pipeline events will be degraded")
    else:
        log.info("Redis connected")

    # Non-fatal parallel startup tasks
    await asyncio.gather(
        _probe_tts(),
        _ensure_qdrant_collections(),
        _preload_embedders(),
        return_exceptions=True,
    )

    # Model preload (can raise — will propagate as warning)
    try:
        await _preload_models()
    except Exception as exc:
        log.warning("model preload warning: %s", exc)

    # Teams meeting pipeline (DB + MinIO + scheduler)
    if _MEETING_PIPELINE_AVAILABLE:
        for label, coro in [
            ("Alembic",    asyncio.to_thread(_run_alembic_upgrade)),
            ("DB tables",  init_db()),
            ("MinIO",      ensure_bucket()),
        ]:
            try:
                await coro
                log.info("%s ready", label)
            except Exception as exc:
                log.warning("%s warning (non-fatal): %s", label, exc)
        try:
            start_scheduler()
        except Exception as exc:
            log.warning("Scheduler warning (non-fatal): %s", exc)

    log.info("=== startup complete ===")
    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    log.info("shutting down…")
    if _MEETING_PIPELINE_AVAILABLE:
        try:
            stop_scheduler()
        except Exception:
            pass
    await close_redis()
    log.info("shutdown complete")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
except RuntimeError as exc:
    log.error("could not mount /static: %s", exc)

# Always-available routes
app.include_router(rest_router)
app.include_router(upload_router)
app.include_router(ws_progress_router)

# Voice assistant routes
if _VOICE_AVAILABLE and ws_router:
    app.include_router(ws_router)

# Teams meeting pipeline routes
if _MEETING_PIPELINE_AVAILABLE:
    app.include_router(webhook_router)
    app.include_router(meetings_router)
    app.include_router(action_router)
