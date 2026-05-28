import asyncio
import logging
import os
import shutil
import subprocess
import uuid
import httpx
import faiss
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.config import OLLAMA_BASE_URL, MEETING_LLM_MODEL, MEETING_LLM_TIMEOUT, UPLOADS_DIR, MEETINGS_INPUT_DIR

log = logging.getLogger("upload")
router = APIRouter(prefix="/upload", tags=["upload"])

UPLOAD_DIR = UPLOADS_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)

# { job_id: {status, progress, filename, transcript_path, summary, rag_index, rag_chunks, error} }
_jobs: dict[str, dict] = {}

UPLOAD_CHUNK         = 4 * 1024 * 1024
_STATUS_UPLOADING    = "uploading"
_STATUS_EXTRACTING   = "extracting_audio"
_STATUS_TRANSCRIBING = "transcribing"
_STATUS_SUMMARIZING  = "summarizing"
_STATUS_DONE         = "done"
_STATUS_FAILED       = "failed"

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _extract_audio(video_path: str, audio_path: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:400]}")


def _transcribe_file(audio_path: str, job_id: str) -> str:
    from faster_whisper import WhisperModel

    try:
        import whisperx
        hf_token = os.getenv("HF_TOKEN")
        model_wx = whisperx.load_model("base", "cpu", language="en")
        audio_wx = whisperx.load_audio(audio_path)
        result   = model_wx.transcribe(audio_wx, batch_size=16)
        language = result.get("language", "en")

        align_model, meta = whisperx.load_align_model(language_code=language, device="cpu")
        result = whisperx.align(result["segments"], align_model, meta, audio_wx, "cpu")

        if hf_token:
            diarize = whisperx.DiarizationPipeline(use_auth_token=hf_token, device="cpu")
            result  = whisperx.assign_word_speakers(diarize(audio_wx), result)

        segs  = result.get("segments", [])
        total = float(segs[-1]["end"]) if segs else 1.0
        lines = []
        for seg in segs:
            speaker = seg.get("speaker", "SPEAKER_00")
            start, end = seg.get("start", 0), seg.get("end", 0)
            lines.append(f"[{start:.1f}s–{end:.1f}s] {speaker}: {seg.get('text','').strip()}")
            _jobs[job_id]["progress"] = min(99, int((end / total) * 100))

        _jobs[job_id]["progress"] = 100
        return "\n".join(lines)

    except ImportError:
        pass
    except Exception as e:
        log.warning(f"[upload] whisperx failed ({e}), falling back to faster-whisper")

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments_gen, info = model.transcribe(
        audio_path, language="en", beam_size=5, vad_filter=True, word_timestamps=False,
    )
    total_dur = float(info.duration) if info.duration else 1.0
    lines = []
    for seg in segments_gen:
        lines.append(f"[{seg.start:.1f}s–{seg.end:.1f}s] {seg.text.strip()}")
        _jobs[job_id]["progress"] = min(99, int((seg.end / total_dur) * 100))
    _jobs[job_id]["progress"] = 100
    return "\n".join(lines)


def _build_rag_index(transcript: str, summary: str = ""):
    lines = [l.strip() for l in transcript.split("\n") if len(l.strip()) > 20]
    if not lines:
        return None, []

    chunks = []
    # Summary as its own searchable chunk so broad questions can match it
    if summary and len(summary.strip()) > 20:
        chunks.append(f"[MEETING SUMMARY]\n{summary.strip()}")

    # Overlapping windows of 6 lines (step 3) for richer context per chunk
    window, step = 6, 3
    for i in range(0, len(lines), step):
        block = "\n".join(lines[i:i + window])
        if block.strip():
            chunks.append(block)

    if not chunks:
        return None, []
    try:
        model = _get_embed_model()
        emb   = model.encode(chunks, normalize_embeddings=True).astype("float32")
        idx   = faiss.IndexFlatIP(emb.shape[1])
        idx.add(emb)
        return idx, chunks
    except Exception as e:
        log.warning(f"[upload] RAG index build failed: {e}")
        return None, []


async def _generate_summary(transcript: str) -> str:
    prompt = (
        "You are a meeting assistant. Summarize this meeting transcript concisely.\n"
        "Include: key topics discussed, decisions made, and action items.\n\n"
        f"TRANSCRIPT:\n{transcript[:6000]}\n\nSUMMARY:"
    )
    try:
        async with httpx.AsyncClient(timeout=MEETING_LLM_TIMEOUT) as client:
            r = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": MEETING_LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 512},
                },
            )
            return r.json().get("response", "").strip() or "Summary unavailable."
    except Exception as e:
        log.warning(f"[upload] Summary generation failed: {e}")
        return "Summary unavailable — LLM not reachable."


async def _run_pipeline(job_id: str, video_path: str, filename: str) -> None:
    audio_path = video_path.rsplit(".", 1)[0] + "_audio.wav"
    txt_path   = video_path.rsplit(".", 1)[0] + ".txt"

    try:
        if _ffmpeg_available():
            _jobs[job_id]["status"] = _STATUS_EXTRACTING
            log.info(f"[upload] {job_id}: extracting audio")
            await asyncio.to_thread(_extract_audio, video_path, audio_path)
            os.unlink(video_path)
            source = audio_path
        else:
            log.warning("[upload] ffmpeg not found — passing file directly to whisper")
            source = video_path

        _jobs[job_id]["status"]   = _STATUS_TRANSCRIBING
        _jobs[job_id]["progress"] = 0
        log.info(f"[upload] {job_id}: transcribing")
        transcript = await asyncio.to_thread(_transcribe_file, source, job_id)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcript: {filename}\n\n{transcript}")

        _jobs[job_id]["status"] = _STATUS_SUMMARIZING
        log.info(f"[upload] {job_id}: generating summary")
        summary = await _generate_summary(transcript)

        # Build RAG after summary so summary is embedded as a chunk too
        rag_index, rag_chunks = await asyncio.to_thread(_build_rag_index, transcript, summary)

        _jobs[job_id].update(
            status=_STATUS_DONE,
            progress=100,
            transcript_path=txt_path,
            summary=summary,
            rag_index=rag_index,
            rag_chunks=rag_chunks,
        )

        # Persist transcript + summary to knowledge base so voice chat can answer meeting questions
        try:
            os.makedirs(MEETINGS_INPUT_DIR, exist_ok=True)
            kb_path = os.path.join(MEETINGS_INPUT_DIR, f"{job_id}.txt")
            with open(kb_path, "w", encoding="utf-8") as f:
                f.write(f"# Meeting: {filename}\n\n## Summary\n{summary}\n\n## Transcript\n{transcript}")
            _jobs[job_id]["kb_path"] = kb_path
            from app.core.agent import invalidate_agent_index, ensure_system_initialized
            invalidate_agent_index()
            await ensure_system_initialized()
            log.info(f"[upload] {job_id}: transcript added to agent knowledge base and index rebuilt")
        except Exception as e:
            log.warning(f"[upload] {job_id}: could not update agent knowledge base: {e}")

        log.info(f"[upload] {job_id}: complete")

    except Exception as e:
        log.exception(f"[upload] {job_id}: pipeline failed")
        _jobs[job_id].update(status=_STATUS_FAILED, error=str(e))

    finally:
        for p in (audio_path, video_path):
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass


class QuestionRequest(BaseModel):
    question: str


@router.post("/transcribe")
async def upload_and_transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    job_id    = str(uuid.uuid4())
    ext       = os.path.splitext(file.filename or "upload.mp4")[1] or ".mp4"
    dest_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")

    _jobs[job_id] = {
        "status": _STATUS_UPLOADING, "progress": 0,
        "filename": file.filename, "transcript_path": None,
        "error": None, "summary": None,
        "rag_index": None, "rag_chunks": None,
    }

    try:
        written = 0
        with open(dest_path, "wb") as out:
            while chunk := await file.read(UPLOAD_CHUNK):
                out.write(chunk)
                written += len(chunk)
        log.info(f"[upload] {job_id}: saved {written / 1e6:.1f} MB")
    except Exception as e:
        _jobs[job_id].update(status=_STATUS_FAILED, error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    background_tasks.add_task(_run_pipeline, job_id, dest_path, file.filename or "upload")
    return {"job_id": job_id, "filename": file.filename}


@router.get("/status/{job_id}")
async def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    result = {
        "job_id":   job_id,
        "status":   job["status"],
        "progress": job.get("progress", 0),
        "filename": job.get("filename"),
        "error":    job.get("error"),
    }
    if job["status"] == _STATUS_DONE:
        result["summary"] = job.get("summary", "")
    return result


@router.get("/transcript/{job_id}")
async def get_transcript(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != _STATUS_DONE:
        raise HTTPException(status_code=400, detail=f"Not ready (status: {job['status']})")
    path = job.get("transcript_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=500, detail="Transcript file missing")
    return FileResponse(path, media_type="text/plain", filename=f"transcript_{job_id}.txt")


@router.post("/ask/{job_id}")
async def ask_question(job_id: str, body: QuestionRequest):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != _STATUS_DONE:
        raise HTTPException(status_code=400, detail="Transcript not ready")

    question   = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")

    rag_index   = job.get("rag_index")
    rag_chunks  = job.get("rag_chunks") or []
    summary_txt = (job.get("summary") or "").strip()

    context_parts = []

    if rag_index is not None and rag_chunks:
        try:
            model = _get_embed_model()
            q_emb = model.encode([question], normalize_embeddings=True).astype("float32")
            k = min(10, len(rag_chunks))
            _, indices = rag_index.search(q_emb, k)
            rag_ctx = "\n\n".join(
                rag_chunks[i] for i in indices[0] if i < len(rag_chunks)
            )
        except Exception as e:
            log.warning(f"[upload] RAG retrieval failed: {e}")
            rag_ctx = "\n\n".join(rag_chunks[:15])

        if summary_txt:
            context_parts.append(f"MEETING SUMMARY:\n{summary_txt}")
        if rag_ctx:
            context_parts.append(f"RELEVANT TRANSCRIPT SECTIONS:\n{rag_ctx}")
    else:
        # Fallback: summary + raw transcript file
        if summary_txt:
            context_parts.append(f"MEETING SUMMARY:\n{summary_txt}")
        path = job.get("transcript_path")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                context_parts.append(f"TRANSCRIPT:\n{f.read(6000)}")

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No meeting content available."

    prompt = (
        "You are a meeting transcript assistant. Your ONLY job is to answer questions strictly based on the meeting transcript and summary provided below.\n"
        "STRICT RULES:\n"
        "- Answer ONLY from the meeting content provided. Do NOT use any outside knowledge.\n"
        "- Do NOT talk about the company, products, or anything not explicitly mentioned in this transcript.\n"
        "- If the answer is not found in the meeting content, respond exactly: 'This was not discussed in the meeting.'\n"
        "- Never make up or infer information that is not clearly stated in the transcript.\n\n"
        f"MEETING CONTENT:\n{context}\n\n"
        f"QUESTION: {question}\n\nANSWER (based only on the meeting content above):"
    )

    try:
        async with httpx.AsyncClient(timeout=MEETING_LLM_TIMEOUT) as client:
            r = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": MEETING_LLM_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 512},
                },
            )
            answer = r.json().get("response", "").strip() or "No answer available."
    except Exception as e:
        log.warning(f"[upload] Q&A failed: {e}")
        answer = "Could not generate answer — LLM not reachable."

    return {"question": question, "answer": answer}


@router.delete("/{job_id}")
async def delete_job(job_id: str):
    job = _jobs.pop(job_id, None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    for key in ("transcript_path", "kb_path"):
        path = job.get(key)
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass

    kb_path = job.get("kb_path")
    if kb_path and not os.path.exists(kb_path):
        # File was removed — invalidate agent index so it rebuilds without this meeting
        try:
            from app.core.agent import invalidate_agent_index
            invalidate_agent_index()
        except Exception:
            pass

    return {"deleted": job_id}
