# Technodysis AI voice assistant

Real-time conversational AI voice assistant — sub-200ms perceived latency, full barge-in interrupt support, pause-triggered backchannelling, and a production-ready WebSocket server.

---

## Quick Start

```bash
# 1. Install all dependencies
pip install -r requirements.txt


# 2. Download ollama model in global .ollama folder
ollama pull llama3.2:3b


# 3. Start server
uvicorn main:app --host 0.0.0.0 --port 8000


# 4. Open the site in browser
http://localhost:8000
# or
http://0.0.0.0:8000/


# 5. Enable to Microphone permission in browser
chrome://flags/#unsafely-treat-insecure-origin-as-secure
# now paste the below urls in the textbox and change option to Enabled. 
'http://0.0.0.0:8000,http://localhost:8000'
```

---

## Architecture

```
Browser mic (PCM16, 64 ms frames via ScriptProcessor)
  │
  │  WebSocket binary
  ▼
Silero VAD  ──────────────────────────────────────────→  Barge-in detect
  │ speech_prob per 32ms frame                            (BARGE_IN_HOLD = 80ms)
  ▼
faster-whisper STT (base, cpu, int8)
  │ text transcript  +  stt_ms
  ▼
Groq LLM (llama-3.1-8b-instant)   ←──  emotion/intent classify (parallel)
  │ streaming tokens  +  llm_first_token_ms
  ▼
Sentence-boundary TTS flusher (every . ! ? ; or 80 chars)
  │ sentence chunks
  ▼
edge-tts (Microsoft Neural, cancellable mid-stream)
  │ MP3 bytes  +  tts_first_chunk_ms
  │
  │  WebSocket binary
  ▼
Browser AudioContext → AudioBufferSourceNode → speaker
```

---

## File Structure

```
project/
├── main.py          Server — WebSocket, VAD, pipeline orchestration
├── agent.py         LLM engine — memory, streaming, emotion/intent, retries
├── stt.py           Whisper helpers — confidence filter, noise list
├── tts.py           TTS dispatcher — edge-tts, XTTS, Piper, cancel support
├── requirements.txt Python dependencies
├── .env             API keys and config
└── static/
    └── index.html   Complete frontend (no build step)
```

---

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required. Get from console.groq.com |
| `TTS_ENGINE` | `auto` | `edge` / `xtts` / `piper` / `auto` |
| `EDGE_VOICE` | `en-US-AriaNeural` | Microsoft Neural voice name |
| `EDGE_RATE` | `+0%` | Speaking rate: `+10%` faster, `-10%` slower |
| `EDGE_RETRY_AFTER` | `30` | Seconds before retrying after transient edge error |
| `WHISPER_LANG` | `en` | STT language, or `auto` for multilingual |
| `STT_CONFIDENCE` | `-0.8` | Reject segments with avg_logprob below this |
| `PIPER_MODEL` | `./models/en_US-lessac-medium.onnx` | Path to Piper voice model |

---

## TTS Voice Options

Set `EDGE_VOICE` in `.env`, or change live via the voice dropdown in the UI:

| Voice | Character |
|---|---|
| `en-US-AriaNeural` | Warm female (default) |
| `en-US-GuyNeural` | Natural male |
| `en-US-JennyNeural` | Friendly female |
| `en-GB-SoniaNeural` | British female |
| `en-GB-RyanNeural` | British male |
| `en-IN-NeerjaNeural` | Indian female |
| `en-IN-PrabhatNeural` | Indian male |

Full list: `python -m edge_tts --list-voices`

---

## Interrupt Latency

The system achieves ~60–150ms perceived interrupt latency:

| Stage | Time | Notes |
|---|---|---|
| Client energy VAD fires | 60ms | After mic frame crosses RMS threshold |
| Audio queue flushed | ~2ms | `AudioBufferSourceNode.stop(0)` + `suspend/resume` |
| Server VAD confirms | 80ms | `BARGE_IN_HOLD = 0.08s` |
| TTS stream aborted | ≤40ms | Per-chunk cancel check in `_edge_synthesize()` |
| **Perceived silence** | **~60ms** | Client flush happens before server confirms |

---

## Backchannelling

The agent speaks brief acknowledgements ("Okay…", "Got it…", "Mm-hmm…") during natural mid-speech pauses. Fires when all four conditions are true:

1. User has spoken ≥ 1.5 s total in the current utterance
2. A mid-speech pause ≥ 0.30 s is detected (user is breathing, not finished)
3. Pause is shorter than 0.75 s (utterance has not ended)
4. No backchannel has been sent this utterance (one per turn max)
5. ≥ 5 seconds since the last backchannel globally (prevents spam)

---

## Groq Retry Policy

`chat_stream()` automatically retries on rate-limit and overload errors:

| Attempt | Wait |
|---|---|
| 1 | 1s |
| 2 | 2s |
| 3 | raises (non-retryable or exhausted) |

Retryable error strings: `429`, `rate limit`, `503`, `service unavailable`, `overloaded`.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves `static/index.html` |
| `GET` | `/health` | Server status, engine, voice, version |
| `GET` | `/session/{id}` | Session memory: emotion, intent, turns, voice |
| `GET` | `/session/{id}/history` | Full conversation as JSON |
| `POST` | `/session/{id}/voice` | Change voice: `{"voice": "en-GB-SoniaNeural"}` |
| `WS` | `/ws` | Main WebSocket endpoint |

---

## WebSocket Protocol

**Server → Client messages:**

| Type | Payload | When |
|---|---|---|
| `session_start` | `{session_id, tts_engine, tts_voice}` | On connect |
| `status` | `{text}` | State changes: idle/listening/thinking/speaking |
| `partial` | `{text}` | Interim transcript during speech |
| `transcript` | `{text, stt_ms}` | Final confirmed transcript + STT latency |
| `ai_stream` | `{text}` | One LLM token |
| `ai_done` | `{session, latency:{stt_ms,llm_ms,tts_ms,total_ms}}` | Turn complete |
| `backchannel` | `{text}` | Filler phrase text (audio follows as binary) |
| `interrupt` | `{reason, gen_id}` | Barge-in confirmed |
| `voice_changed` | `{voice}` | Voice hot-swap confirmed |
| `error` | `{text}` | Pipeline error (agent returns to idle) |
| `server_ping` | — | Keepalive (respond with `{type:"pong"}`) |
| `reset_ack` | — | Session cleared |
| Binary | MP3 bytes | TTS audio chunk |

**Client → Server messages:**

| Type | Payload | Action |
|---|---|---|
| Binary PCM16 | 1024 samples @ 16kHz | Audio frame for VAD + recording |
| `interrupt` | — | Client-initiated barge-in |
| `ping` | — | Heartbeat, returns `pong` + session state |
| `pong` | — | Response to `server_ping` |
| `set_voice` | `{voice}` | Change TTS voice mid-session |
| `reset` | — | Clear conversation history |

---

## Tuning Constants

All in `main.py` near the top:

| Constant | Default | Effect |
|---|---|---|
| `SPEECH_THRESHOLD` | `0.60` | Lower for noisy rooms |
| `SILENCE_LIMIT` | `0.75s` | Increase if users get cut off |
| `BARGE_IN_HOLD` | `0.08s` | Increase to reduce false interrupts |
| `BC_MIN_SPEECH_BEFORE` | `1.50s` | Minimum speech before a backchannel |
| `BC_PAUSE_THRESHOLD` | `0.30s` | Pause length to trigger backchannel |
| `TTS_FLUSH_CHARS` | `80` | Max chars before forcing a TTS flush |
| `WS_PING_INTERVAL` | `25s` | Keepalive ping frequency |
| `WS_PING_MAX_MISSED` | `2` | Close connection after this many missed pongs |

In `index.html`:

| Constant | Default | Effect |
|---|---|---|
| `CLIENT_RMS_THRESH` | `18` | Raise if background noise causes false barge-in |
| `CLIENT_BARGE_HOLD` | `60ms` | Client-side interrupt confirmation hold time |

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `Space` | Interrupt the assistant (while speaking) |
| `R` | Reset conversation (while idle) |
| `H` | Toggle conversation history panel |

---

## Known Limitations

- `ScriptProcessor` is deprecated in Chrome — future migration to `AudioWorklet` recommended for long-term support
- No authentication on the WebSocket endpoint — add an API key header check for production
- Whisper runs synchronously on CPU; GPU acceleration requires `device="cuda"` in `stt.py` and `main.py`
- `edge-tts` requires outbound HTTPS to Microsoft servers — will not work in fully air-gapped environments (use Piper instead)
- Single-server only — no Redis session store for horizontal scaling
