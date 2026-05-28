import asyncio
import logging
import os
import shutil
import subprocess
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse, JSONResponse, FileResponse

log = logging.getLogger("upload")
router = APIRouter(prefix="/upload", tags=["upload"])

# ── Storage ───────────────────────────────────────────────────────────────────
_BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
UPLOAD_DIR = os.path.join(_BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── In-memory job registry ────────────────────────────────────────────────────
# { job_id: {status, progress, filename, transcript_path, error} }
_jobs: dict[str, dict] = {}

_STATUS_UPLOADING    = "uploading"
_STATUS_EXTRACTING   = "extracting_audio"
_STATUS_TRANSCRIBING = "transcribing"
_STATUS_DONE         = "done"
_STATUS_FAILED       = "failed"

UPLOAD_CHUNK = 4 * 1024 * 1024   # 4 MB read chunks — safe for large files


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _extract_audio(video_path: str, audio_path: str) -> None:
    """Use ffmpeg to extract 16kHz mono WAV from video (much smaller than raw MP4)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                          # no video
        "-acodec", "pcm_s16le",         # PCM WAV
        "-ar", "16000",                 # 16 kHz — whisper's native rate
        "-ac", "1",                     # mono
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:400]}")


def _transcribe_file(audio_path: str, job_id: str) -> str:
    """
    Run faster-whisper on audio_path.
    Updates _jobs[job_id]['progress'] (0-100) as segments are processed.
    Returns formatted transcript string.
    """
    from faster_whisper import WhisperModel  # type: ignore

    # Try whisperx first for speaker diarization
    try:
        import whisperx  # type: ignore
        import os as _os
        hf_token = _os.getenv("HF_TOKEN")
        model_wx = whisperx.load_model("base", "cpu", language="en")
        audio_wx = whisperx.load_audio(audio_path)
        result   = model_wx.transcribe(audio_wx, batch_size=16)
        language = result.get("language", "en")

        align_model, meta = whisperx.load_align_model(language_code=language, device="cpu")
        result = whisperx.align(result["segments"], align_model, meta, audio_wx, "cpu")

        if hf_token:
            diarize = whisperx.DiarizationPipeline(use_auth_token=hf_token, device="cpu")
            dsegs   = diarize(audio_wx)
            result  = whisperx.assign_word_speakers(dsegs, result)

        segs  = result.get("segments", [])
        total = float(segs[-1]["end"]) if segs else 1.0
        lines = []
        for seg in segs:
            speaker = seg.get("speaker", "SPEAKER_00")
            start   = seg.get("start", 0)
            end     = seg.get("end", 0)
            text    = seg.get("text", "").strip()
            lines.append(f"[{start:.1f}s–{end:.1f}s] {speaker}: {text}")
            _jobs[job_id]["progress"] = min(99, int((end / total) * 100))

        _jobs[job_id]["progress"] = 100
        return "\n".join(lines)

    except ImportError:
        pass  # whisperx not installed — fall through to faster-whisper
    except Exception as e:
        log.warning(f"[upload] whisperx failed ({e}), falling back to faster-whisper")

    # ── faster-whisper fallback (no speaker labels) ───────────────────────────
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments_gen, info = model.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        vad_filter=True,        # skip silent regions — faster on long files
        word_timestamps=False,
    )

    total_duration = float(info.duration) if info.duration else 1.0
    lines: list[str] = []

    for seg in segments_gen:
        start = seg.start
        end   = seg.end
        text  = seg.text.strip()
        lines.append(f"[{start:.1f}s–{end:.1f}s] {text}")
        _jobs[job_id]["progress"] = min(99, int((end / total_duration) * 100))

    _jobs[job_id]["progress"] = 100
    return "\n".join(lines)


async def _run_pipeline(job_id: str, video_path: str, filename: str) -> None:
    """Full background pipeline: extract audio → transcribe → save → cleanup."""
    audio_path = video_path.rsplit(".", 1)[0] + "_audio.wav"
    txt_path   = video_path.rsplit(".", 1)[0] + ".txt"

    try:
        # Step 1 — audio extraction
        if _ffmpeg_available():
            _jobs[job_id]["status"] = _STATUS_EXTRACTING
            log.info(f"[upload] {job_id}: extracting audio with ffmpeg")
            await asyncio.to_thread(_extract_audio, video_path, audio_path)
            # Video no longer needed — free disk space
            os.unlink(video_path)
            source = audio_path
        else:
            log.warning("[upload] ffmpeg not found — passing video directly to whisper (slower)")
            source = video_path

        # Step 2 — transcription (runs in thread so event loop stays free)
        _jobs[job_id]["status"]   = _STATUS_TRANSCRIBING
        _jobs[job_id]["progress"] = 0
        log.info(f"[upload] {job_id}: starting transcription")
        transcript = await asyncio.to_thread(_transcribe_file, source, job_id)

        # Step 3 — save
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcript: {filename}\n\n")
            f.write(transcript)

        _jobs[job_id].update(
            status=_STATUS_DONE,
            progress=100,
            transcript_path=txt_path,
        )
        log.info(f"[upload] {job_id}: done")

    except Exception as e:
        log.exception(f"[upload] {job_id}: pipeline failed")
        _jobs[job_id].update(status=_STATUS_FAILED, error=str(e))

    finally:
        # Clean up audio/video files to free disk space
        for p in (audio_path, video_path):
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/transcribe")
async def upload_and_transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accept a large video file, stream it to disk in 4MB chunks,
    and immediately return a job_id. Processing runs in the background.
    Poll GET /upload/status/{job_id} for progress.
    """
    job_id   = str(uuid.uuid4())
    ext      = os.path.splitext(file.filename or "upload.mp4")[1] or ".mp4"
    dest_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")

    _jobs[job_id] = {
        "status":          _STATUS_UPLOADING,
        "progress":        0,
        "filename":        file.filename,
        "transcript_path": None,
        "error":           None,
    }

    # Stream to disk in 4MB chunks — never buffers the full file in memory
    try:
        written = 0
        with open(dest_path, "wb") as out:
            while True:
                chunk = await file.read(UPLOAD_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
        log.info(f"[upload] {job_id}: saved {written / 1e6:.1f} MB → {dest_path}")
    except Exception as e:
        _jobs[job_id].update(status=_STATUS_FAILED, error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    # Kick off background pipeline
    background_tasks.add_task(_run_pipeline, job_id, dest_path, file.filename or "upload")

    return {"job_id": job_id, "filename": file.filename}


@router.get("/status/{job_id}")
async def job_status(job_id: str):
    """Return current status + progress percentage (0-100)."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id":   job_id,
        "status":   job["status"],
        "progress": job.get("progress", 0),
        "filename": job.get("filename"),
        "error":    job.get("error"),
    }


@router.get("/transcript/{job_id}")
async def get_transcript(job_id: str):
    """Return the plain-text transcript once status == done."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != _STATUS_DONE:
        raise HTTPException(status_code=400, detail=f"Not ready (status: {job['status']})")

    path = job.get("transcript_path")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=500, detail="Transcript file missing")

    return FileResponse(path, media_type="text/plain", filename=f"transcript_{job_id}.txt")


@router.delete("/{job_id}")
async def delete_job(job_id: str):
    """Clean up transcript file and job record."""
    job = _jobs.pop(job_id, None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    path = job.get("transcript_path")
    if path and os.path.exists(path):
        os.unlink(path)

    return {"deleted": job_id}
