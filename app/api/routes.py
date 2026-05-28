"""
Defines FastAPI routes that handle all incoming HTTP requests.

- Returns the frontend UI file (index.html) when user opens the app
- Gives a health API to check if Whisper, VAD, and TTS are working
- Provides API to get session details using session_id
- Provides API to get past conversation history of a session
- Allows changing the TTS voice for a session after validation
- Connects these APIs with agent logic and voice/TTS modules

This file is the main connection between frontend requests and backend logic.
"""


import logging
import httpx
from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from app.core.voices import is_supported_voice
from app.core.tts import get_engine_name, get_edge_voice
from app.core.agent import get_session_state, get_conversation_history, update_session_voice
from app.config import STATIC_DIR, UI_FILE_NAME, HEALTH_STT_LABEL, HEALTH_VAD_LABEL, APP_VERSION, OLLAMA_BASE_URL


log = logging.getLogger("routes")
router = APIRouter()


class VoiceRequest(BaseModel):
    voice: str = Field(..., min_length=1)


@router.get("/")
async def root():
    try:
        return FileResponse(f"{STATIC_DIR}/{UI_FILE_NAME}")
    except Exception as e:
        log.error(f"[/] Failed to serve {UI_FILE_NAME}: {e}")
        return JSONResponse({"error": "Frontend not found."}, status_code=404)


@router.get("/health")
async def health():
    try:
        from app.db.database import check_db_health
        from app.core.storage import check_minio_health

        db_ok    = False
        minio_ok = False
        ollama_ok = False

        try:
            db_ok = await check_db_health()
        except Exception:
            pass

        try:
            minio_ok = await check_minio_health()
        except Exception:
            pass

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                ollama_ok = r.status_code == 200
        except Exception:
            pass

        overall = "ok" if all([db_ok, minio_ok, ollama_ok]) else "degraded"

        return {
            "status":  overall,
            "whisper": HEALTH_STT_LABEL,
            "vad":     HEALTH_VAD_LABEL,
            "tts":     get_engine_name(),
            "voice":   get_edge_voice(),
            "version": APP_VERSION,
            "services": {
                "database": "ok" if db_ok    else "error",
                "minio":    "ok" if minio_ok  else "error",
                "ollama":   "ok" if ollama_ok else "error",
            },
        }
    except Exception as e:
        log.error(f"[/health] Error: {e}")
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


@router.get("/session/{session_id}")
async def session_info(session_id: str):
    try:
        return JSONResponse(get_session_state(session_id))
    except Exception as e:
        log.error(f"[/session/{session_id}] Error: {e}")
        return JSONResponse({"error": "Failed to retrieve session state."}, status_code=500)


@router.get("/session/{session_id}/history")
async def session_history(session_id: str):
    try:
        return JSONResponse({
            "session_id": session_id,
            "history": get_conversation_history(session_id),
        })
    except Exception as e:
        log.error(f"[/session/{session_id}/history] Error: {e}")
        return JSONResponse({"error": "Failed to retrieve history."}, status_code=500)


@router.post("/session/{session_id}/voice")
async def session_set_voice(session_id: str, body: VoiceRequest):
    try:
        voice = body.voice.strip()
        if not is_supported_voice(voice):
            return JSONResponse({"error": "Invalid voice"}, status_code=400)
        update_session_voice(session_id, voice)
        return JSONResponse({"session_id": session_id, "voice": voice})
    except Exception as e:
        log.error(f"[/session/{session_id}/voice] Error: {e}")
        return JSONResponse({"error": "Failed to set voice."}, status_code=500)

