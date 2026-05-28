"""
Main file that starts the Voice Agent application.

- Creates FastAPI server and sets up logging
- Loads API routes and WebSocket connection
- Serves frontend files

- At startup:
  • checks if TTS is working
  • loads STT, Agent, and Speaker models

- Makes sure everything is ready before users connect

This file starts and prepares the whole system to run.
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import asyncio
import logging
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.routes import router as rest_router
from app.api.upload import router as upload_router
from app.config import STATIC_DIR, LOGS_DIR, LOG_FILE, LOG_MAX_BYTES, LOG_BACKUP_COUNT, APP_VERSION, APP_TITLE

# ── Optional: voice assistant (needs full requirements.txt) ───────────────────
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
    import logging as _log
    _log.getLogger("main").warning(
        "Voice assistant disabled (missing packages: %s). "
        "Video upload + transcription still fully available.", _e
    )

# ── Optional: Teams meeting pipeline (needs DB + full requirements) ───────────
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
from warnings import filterwarnings


filterwarnings("ignore")


def _run_alembic_upgrade():
    """Run 'alembic upgrade head' in a thread at startup — safe to call repeatedly."""
    from alembic.config import Config
    from alembic import command
    cfg = Config(os.path.join(os.path.dirname(__file__), "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(os.path.dirname(__file__), "alembic"))
    command.upgrade(cfg, "head")

# Logging setup
os.makedirs(LOGS_DIR, exist_ok=True)

# Suppress noisy third-party loggers
logging.getLogger("speechbrain").setLevel(logging.WARNING)
logging.getLogger("speechbrain.utils.fetching").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_fmt = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_console = logging.StreamHandler()
_console.setFormatter(_fmt)

_file = RotatingFileHandler(
    LOG_FILE,
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT,
    encoding="utf-8",
)
_file.setFormatter(_fmt)

logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(_console)
logging.getLogger().addHandler(_file)

log = logging.getLogger("voice-agent")



async def _probe_tts():
    if not _VOICE_AVAILABLE:
        return
    log.info(f"[TTS] startup probe — voice={get_edge_voice()!r}")
    try:
        data = await synthesize_chunks("Hello.")
        if data:
            log.info(f"[TTS] OK — engine={get_engine_name()} {len(data)}B")
        else:
            log.warning("[TTS] 0 bytes returned — voice responses may not work")
    except Exception as e:
        log.warning(f"[TTS] Startup probe failed (non-fatal): {e}")


async def _preload_core_models():
    if not _VOICE_AVAILABLE:
        log.info("[startup] Slim mode — skipping voice assistant model preload")
        return

    log.info("[startup] Preloading STT, Agent, and Speaker models...")

    stt_result, agent_result, speaker_result = await asyncio.gather(
        asyncio.to_thread(get_model),
        ensure_system_initialized(),
        asyncio.to_thread(get_classifier),
        return_exceptions=True,
    )

    if isinstance(stt_result, Exception):
        raise RuntimeError(f"STT preload failed: {stt_result}")
    log.info("[startup] STT ready")

    if isinstance(agent_result, Exception):
        log.warning("[startup] Ollama unavailable — voice assistant disabled (%s)", agent_result)
    else:
        log.info("[startup] Agent/RAG/Ollama ready")

    if isinstance(speaker_result, Exception):
        log.warning("[startup] Speaker model not loaded (non-fatal): %s", speaker_result)
    else:
        log.info("[startup] Speaker model ready")



# Lifespan manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):

    log.info("[startup] Voice agent starting up...")

    try:
        await _probe_tts()
    except Exception as e:
        log.warning("[startup] TTS probe failed (non-fatal — voice responses disabled): %s", e)

    try:
        await _preload_core_models()
        log.info("[startup] Core models ready")
    except Exception as e:
        log.exception("[startup] Fatal startup error: %s", e)
        raise

    # Optional: Teams meeting pipeline (DB + MinIO + scheduler)
    if _MEETING_PIPELINE_AVAILABLE:
        try:
            await asyncio.to_thread(_run_alembic_upgrade)
            log.info("[startup] Alembic migrations applied")
        except Exception as e:
            log.warning("[startup] Alembic warning (non-fatal): %s", e)

        try:
            await init_db()
            log.info("[startup] Database tables ensured")
        except Exception as e:
            log.warning("[startup] DB init warning (non-fatal): %s", e)

        try:
            await ensure_bucket()
            log.info("[startup] MinIO bucket ensured")
        except Exception as e:
            log.warning("[startup] MinIO warning (non-fatal): %s", e)

        try:
            start_scheduler()
        except Exception as e:
            log.warning("[startup] Scheduler warning (non-fatal): %s", e)

    yield

    log.info("[shutdown] Shutting down...")
    if _MEETING_PIPELINE_AVAILABLE:
        stop_scheduler()



# FastAPI app initialization with CORS and routes
app = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
try:
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
except RuntimeError as e:
    log.error(f"[startup] Could not mount /static: {e}")

# Core routes (always available)
app.include_router(rest_router)
app.include_router(upload_router)

# Voice assistant routes (only when full packages are installed)
if _VOICE_AVAILABLE and ws_router:
    app.include_router(ws_router)

# Teams meeting pipeline routes (only when DB/full packages are installed)
if _MEETING_PIPELINE_AVAILABLE:
    app.include_router(webhook_router)
    app.include_router(meetings_router)
    app.include_router(action_router)

