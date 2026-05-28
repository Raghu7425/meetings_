"""
Handles text-to-speech (TTS) conversion using Edge TTS.

- Converts AI text into audio output
- Cleans text before speech (removes symbols, unwanted characters)

- Supports emotion-based voice changes:
  • adjusts rate, pitch, and volume

- Generates audio in chunks for streaming
- Supports parallel processing for multiple segments
- Allows changing and retrieving current voice

- Provides helper functions to:
  • get available voices
  • get current voice and engine name

This file converts AI responses into natural-sounding speech.
"""


import io
import asyncio
import logging
import threading
import re
from app.config import TTS_VOICE, TTS_RATE, TTS_VOLUME, TTS_PITCH


log = logging.getLogger("tts")

_current_voice = TTS_VOICE
_voice_lock    = threading.Lock()


# Emotion → SSML prosody overrides
EMOTION_PRESETS: dict[str, dict] = {
    "happy":     {"rate": "+15%", "pitch": "+3Hz",  "volume": "+5%"},
    "excited":   {"rate": "+20%", "pitch": "+5Hz",  "volume": "+8%"},
    "sad":       {"rate": "-15%", "pitch": "-3Hz",  "volume": "-5%"},
    "angry":     {"rate": "+10%", "pitch": "+2Hz",  "volume": "+10%"},
    "fearful":   {"rate": "+5%",  "pitch": "+1Hz",  "volume": "-3%"},
    "surprised": {"rate": "+18%", "pitch": "+4Hz",  "volume": "+5%"},
    "neutral":   {"rate": "+0%",  "pitch": "+0Hz",  "volume": "+0%"},
    "calm":      {"rate": "-10%", "pitch": "-2Hz",  "volume": "-5%"},
}



def _text_clean(text: str) -> str:

    if not text:
        return ""
    
    try:
        text = re.sub(r"\*+", "", text)
        text = re.sub(r"#+\s*", "", text)
        text = re.sub(r"`+", "", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = (text
                .replace("\u2192", " to ").replace("\u2190", " from ")
                .replace("\u2022", "").replace("\u00b7", "")
                .replace("\u2014", ", ").replace("\u2013", ", ")
                .replace("\u201c", "").replace("\u201d", "")
                .replace("\u2018", "").replace("\u2019", "'"))
        return re.sub(r"\s+", " ", text).strip()
    
    except Exception as e:
        log.error(f"[TTS] _text_clean error: {e}")
        return text.strip() if isinstance(text, str) else ""



# Core synthesis
async def _synthesize_edge(text: str, 
                           voice:  str | None = None, 
                           rate:   str | None = None, 
                           volume: str | None = None, 
                           pitch:  str | None = None,
) -> bytes:
    
    cleaned = _text_clean(text)
    
    if not cleaned:
        return b""

    try:
        import edge_tts
    except ImportError:
        log.error("[TTS] edge-tts not installed. Run: pip install edge-tts")
        return b""

    try:
        with _voice_lock:
            v = voice or _current_voice
    except Exception as e:
        log.error(f"[TTS] Failed to acquire voice lock: {e}")
        v = TTS_VOICE

    r  = rate   or TTS_RATE
    vo = volume or TTS_VOLUME
    p  = pitch  or TTS_PITCH

    try:
        communicate = edge_tts.Communicate(cleaned, voice=v, rate=r, volume=vo, pitch=p)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        data = buf.read()
        log.debug(f"[TTS] {len(data)}B voice={v} rate={r}: {cleaned!r}")
        return data

    except asyncio.CancelledError:
        raise
    except Exception as e:
        log.error(f"[TTS] edge-tts generation error: {e}")
        return b""



# Parallel multi-segment synthesis
async def _synthesize_parallel(segments: list[dict]) -> bytes:

    if not segments:
        return b""

    async def _render(seg: dict) -> bytes:  
        try:
            emotion = seg.get("emotion", "neutral")
            if not isinstance(emotion, str):
                emotion = "neutral"
            emotion = emotion.lower()
            preset  = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS["neutral"])
        
            return await _synthesize_edge(
                text   = seg.get("text", ""),
                rate   = seg.get("rate",   preset["rate"]),
                volume = seg.get("volume", preset["volume"]),
                pitch  = seg.get("pitch",  preset["pitch"]),
            )
        
        except asyncio.CancelledError:
            raise
        
        except Exception as e:
            log.error(f"[TTS] Error rendering segment {seg!r}: {e}")
            return b""

    try:
        results = await asyncio.gather(*[_render(seg) for seg in segments])
    
    except asyncio.CancelledError:
        raise
    
    except Exception as e:
        log.error(f"[TTS] gather error: {e}")
        return b""

    return b"".join(r for r in results if isinstance(r, bytes) and r)



# Public interface
async def synthesize_chunks(text: str, 
                            cancel: asyncio.Event | None = None, 
                            voice: str | None = None, 
                            segments: list[dict] | None = None) -> bytes:
    try:
        if cancel and cancel.is_set():
            return b""
            
        if segments:
            if not isinstance(segments, list):
                log.warning(f"[TTS] segments must be a list, got {type(segments)}. Ignoring.")
            else:
                return await _synthesize_parallel(segments)
        
        if not text or not text.strip():
            return b""
        
        if not isinstance(text, str):
            log.warning(f"[TTS] synthesize_chunks received non-string: {type(text)}")
            return b""
        
        return await _synthesize_edge(text, voice=voice)
    
    except asyncio.CancelledError:
        raise
    
    except Exception as e:
        log.error(f"[TTS] Unexpected error in synthesize_chunks: {e}")
        return b""



async def synthesize(text: str, cancel: asyncio.Event | None = None) -> bytes:
    return await synthesize_chunks(text, cancel)



def synthesize_sync(text: str) -> bytes:
    try:
        return asyncio.run(_synthesize_edge(text))
    
    except RuntimeError as e:
        log.error(f"[TTS] synthesize_sync called inside running event loop: {e}")
        return b""
    
    except Exception as e:
        log.error(f"[TTS] synthesize_sync error: {e}")
        return b""



# Voice management
def set_edge_voice(voice: str) -> None:

    global _current_voice
    
    if not voice or not isinstance(voice, str):
        log.warning(f"[TTS] set_edge_voice called with invalid value: {voice!r}")
        return
    
    try:
        with _voice_lock:
            _current_voice = voice
        log.info(f"[TTS] voice set to: {voice!r}")
    
    except Exception as e:
        log.error(f"[TTS] Failed to set voice: {e}")



def get_edge_voice() -> str:
    try:
        with _voice_lock:
            return _current_voice
    
    except Exception as e:
        log.error(f"[TTS] Failed to read current voice: {e}")
        return TTS_VOICE



def get_engine_name() -> str:
    return "edge-tts"


def get_emotion_presets() -> dict:
    return dict(EMOTION_PRESETS)



async def get_available_voices() -> list[str]:
    try:
        import edge_tts
    
    except ImportError:
        log.error("[TTS] edge-tts not installed.")
        return [TTS_VOICE]
    
    try:
        voices = await edge_tts.list_voices()
        if not voices:
            return [TTS_VOICE]
        return [v["ShortName"] for v in voices if "ShortName" in v]
    
    except Exception as e:
        log.error(f"[TTS] Could not list voices: {e}")
        return [TTS_VOICE]

