"""
Upload API — Redis Streams-backed event-driven pipeline.

Flow
────
  POST /upload/transcribe
      ├─ Save file to disk
      ├─ Initialize job state in Redis (RedisJobStore)
      ├─ Publish QUEUED event to Redis Stream
      └─ Launch PipelineRunner as asyncio background task

  GET /upload/status/{job_id}
      └─ Read job hash from Redis (real-time state)

  GET /upload/transcript/{job_id}
      └─ Read saved .txt file from disk

  POST /upload/ask/{job_id}
      ├─ Try Qdrant (job-scoped hybrid search)
      ├─ Fallback: FAISS in-memory index (legacy)
      └─ LLM answer generation with retry

  DELETE /upload/{job_id}
      ├─ Delete Redis job hash
      ├─ Delete Qdrant vectors for job
      └─ Delete transcript file from disk

WebSocket progress endpoint lives in ws_progress.py.
"""

from __future__ import annotations

import json
import logging
import os
import uuid

import httpx
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import (
    MEETINGS_INPUT_DIR,
    MEETING_LLM_MODEL,
    MEETING_LLM_TIMEOUT,
    OLLAMA_BASE_URL,
    UPLOADS_DIR,
)
from app.core.pipeline.event_bus import PipelineEvent, publish_event
from app.core.pipeline.stages import PipelineRunner
from app.db.redis_client import RedisJobStore, get_redis
from app.utils.retry import retry_async

log = logging.getLogger("upload")
router = APIRouter(prefix="/upload", tags=["upload"])

os.makedirs(UPLOADS_DIR, exist_ok=True)

UPLOAD_CHUNK = 4 * 1024 * 1024  # 4 MB streaming read buffer


# ── POST /upload/transcribe ────────────────────────────────────────────────────

@router.post("/transcribe")
async def upload_and_transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    job_id    = str(uuid.uuid4())
    ext       = os.path.splitext(file.filename or "upload.mp4")[1] or ".mp4"
    dest_path = os.path.join(UPLOADS_DIR, f"{job_id}{ext}")

    r = await get_redis()
    store = RedisJobStore(r)

    await store.set(job_id, {
        "status":          PipelineEvent.QUEUED.value,
        "progress":        "0",
        "filename":        file.filename or "upload",
        "transcript_path": "",
        "summary":         "",
        "error":           "",
    })

    # Stream file to disk in chunks to avoid loading entire file into memory
    try:
        written = 0
        with open(dest_path, "wb") as out:
            while chunk := await file.read(UPLOAD_CHUNK):
                out.write(chunk)
                written += len(chunk)
        log.info("upload saved job=%s size_mb=%.1f", job_id, written / 1e6)
    except Exception as exc:
        await store.update(job_id, status=PipelineEvent.FAILED.value, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}")

    await publish_event(job_id, PipelineEvent.QUEUED, progress=0, message="Upload complete — queued")

    # Launch pipeline as a non-blocking background task
    runner = PipelineRunner(job_id, dest_path, file.filename or "upload")
    background_tasks.add_task(runner.run)

    return {"job_id": job_id, "filename": file.filename}


# ── GET /upload/status/{job_id} ────────────────────────────────────────────────

@router.get("/status/{job_id}")
async def job_status(job_id: str):
    r = await get_redis()
    store = RedisJobStore(r)
    data = await store.get(job_id)
    if not data:
        # Fallback: check legacy in-memory dict (imported lazily to avoid circular import)
        raise HTTPException(status_code=404, detail="Job not found")

    result: dict = {
        "job_id":   job_id,
        "status":   data.get("status", "unknown"),
        "progress": int(data.get("progress", 0)),
        "filename": data.get("filename", ""),
        "error":    data.get("error", "") or None,
    }

    if data.get("status") == PipelineEvent.DONE.value:
        result["summary"] = data.get("summary", "")
        raw_sd = data.get("structured_data", "")
        if raw_sd:
            try:
                result["structured_data"] = json.loads(raw_sd)
            except json.JSONDecodeError:
                result["structured_data"] = None

    return result


# ── GET /upload/transcript/{job_id} ───────────────────────────────────────────

@router.get("/transcript/{job_id}")
async def get_transcript(job_id: str):
    r = await get_redis()
    store = RedisJobStore(r)
    data = await store.get(job_id)

    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    if data.get("status") != PipelineEvent.DONE.value:
        raise HTTPException(status_code=400, detail=f"Not ready (status: {data.get('status')})")

    path = data.get("transcript_path", "")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=500, detail="Transcript file missing")

    return FileResponse(path, media_type="text/plain", filename=f"transcript_{job_id}.txt")


# ── POST /upload/ask/{job_id} ──────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str


@router.post("/ask/{job_id}")
async def ask_question(job_id: str, body: QuestionRequest):
    r = await get_redis()
    store = RedisJobStore(r)
    data = await store.get(job_id)

    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    if data.get("status") != PipelineEvent.DONE.value:
        raise HTTPException(status_code=400, detail="Transcript not ready")

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")

    context_parts: list[str] = []
    summary_txt = (data.get("summary") or "").strip()

    # ── Qdrant hybrid search (primary path) ──
    qdrant_ok = False
    try:
        from app.core.vector_store.qdrant_store import get_qdrant_store
        from app.core.vector_store.embedder import MultiLevelEmbedder

        qdrant = get_qdrant_store()
        if qdrant.available:
            embedder = MultiLevelEmbedder()
            hits = await qdrant.search_hybrid(question, job_id, embedder, top_k=10)
            if hits:
                if summary_txt:
                    context_parts.append(f"MEETING SUMMARY:\n{summary_txt}")
                context_parts.append(
                    "RELEVANT SECTIONS:\n\n" +
                    "\n\n".join(f"[{h['section'].upper()}] {h['text']}" for h in hits)
                )
                qdrant_ok = True
    except Exception as exc:
        log.warning("qdrant search failed for ask job=%s: %s", job_id, exc)

    # ── FAISS fallback (legacy in-memory index) ──
    if not qdrant_ok:
        try:
            from app.api._legacy_rag import get_job_rag_index
            idx, chunks = get_job_rag_index(job_id)
            if idx and chunks:
                from app.core.vector_store.embedder import MultiLevelEmbedder
                embedder = MultiLevelEmbedder()
                q_emb = (await embedder.embed_for_query(question))[0].tolist()
                import faiss, numpy as np
                q_arr = np.array([q_emb], dtype="float32")
                k = min(10, len(chunks))
                _, indices = idx.search(q_arr, k)
                rag_ctx = "\n\n".join(chunks[i] for i in indices[0] if i < len(chunks))
                if summary_txt:
                    context_parts.append(f"MEETING SUMMARY:\n{summary_txt}")
                if rag_ctx:
                    context_parts.append(f"RELEVANT TRANSCRIPT SECTIONS:\n{rag_ctx}")
        except Exception as exc:
            log.warning("faiss fallback failed for ask job=%s: %s", job_id, exc)

    # ── Last resort: raw transcript file ──
    if not context_parts:
        if summary_txt:
            context_parts.append(f"MEETING SUMMARY:\n{summary_txt}")
        path = data.get("transcript_path", "")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                context_parts.append(f"TRANSCRIPT:\n{f.read(6000)}")

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No meeting content available."

    prompt = (
        "You are a meeting transcript assistant. Answer questions strictly from the meeting content below.\n"
        "RULES:\n"
        "- Answer ONLY from the content provided. Do NOT use outside knowledge.\n"
        "- If the answer is not in the content, respond: 'This was not discussed in the meeting.'\n"
        "- Never fabricate or infer information not clearly stated.\n\n"
        f"MEETING CONTENT:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER:"
    )

    async def _llm_call() -> str:
        async with httpx.AsyncClient(timeout=MEETING_LLM_TIMEOUT) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model":   MEETING_LLM_MODEL,
                    "prompt":  prompt,
                    "stream":  False,
                    "options": {"temperature": 0.1, "num_predict": 512},
                },
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip() or "No answer available."

    try:
        answer = await retry_async(_llm_call, max_attempts=3)
    except Exception as exc:
        log.warning("Q&A LLM failed job=%s: %s", job_id, exc)
        answer = "Could not generate answer — LLM not reachable."

    return {"question": question, "answer": answer}


# ── DELETE /upload/{job_id} ───────────────────────────────────────────────────

@router.delete("/{job_id}")
async def delete_job(job_id: str):
    r = await get_redis()
    store = RedisJobStore(r)
    data = await store.get(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")

    await store.delete(job_id)

    # Remove transcript file
    path = data.get("transcript_path", "")
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass

    # Remove knowledge-base file
    kb_path = os.path.join(MEETINGS_INPUT_DIR, f"{job_id}.txt")
    if os.path.exists(kb_path):
        try:
            os.unlink(kb_path)
        except OSError:
            pass

    # Remove Qdrant vectors
    try:
        from app.core.vector_store.qdrant_store import get_qdrant_store
        qdrant = get_qdrant_store()
        if qdrant.available:
            await qdrant.delete_meeting(job_id)
    except Exception as exc:
        log.warning("qdrant delete failed job=%s: %s", job_id, exc)

    # Invalidate voice agent index
    try:
        from app.core.agent import invalidate_agent_index
        invalidate_agent_index()
    except Exception:
        pass

    return {"deleted": job_id}
