"""
Handles audio noise reduction and filtering.

- Reads configuration from environment (enable, strength, thresholds)
- Cleans incoming audio by reducing background noise

- Uses noisereduce library if available
- Falls back to simple spectral method if library is not installed

- Selects noise sample from audio and removes it from signal
- Converts audio safely to float format before processing

- Detects low-energy (noise-only) audio and drops it
- Tracks statistics like processed and dropped audio chunks

This file improves audio quality before further processing.
"""


import logging
import numpy as np
from app.config import (NOISE_CANCEL_ENABLED, NOISE_CANCEL_STRENGTH, NOISE_RMS_FLOOR, 
                        NOISE_SAMPLE_SECS, NOISE_FRAME_SIZE, NOISE_HOP_SIZE, 
                        NOISE_QUIET_FRAME_RATIO, NOISE_ALPHA_BASE, NOISE_ALPHA_SCALE, 
                        NOISE_NORM_EPS, SAMPLE_RATE)


log = logging.getLogger("noise_cancel")

_stats = {"processed": 0, "dropped": 0}
_nr_module = None


def _get_noisereduce():

    global _nr_module
    
    if _nr_module is not None:
        return _nr_module

    try:
        import noisereduce as nr
        _nr_module = nr
        log.info("[NC] noisereduce backend ready")
        return _nr_module
    
    except ImportError:
        log.warning("[NC] noisereduce not installed, using spectral fallback")
        _nr_module = False
        return None



def _rms(pcm: np.ndarray) -> float:
    if pcm is None or len(pcm) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(pcm.astype(np.float64)))))



def _safe_float32_pcm(pcm: np.ndarray) -> np.ndarray | None:

    if pcm is None or len(pcm) == 0:
        return pcm

    if not isinstance(pcm, np.ndarray):
        log.warning(f"[NC] expected np.ndarray, got {type(pcm)}")
        return None

    try:
        if pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)
    except Exception as e:
        log.error(f"[NC] float32 cast failed: {e}")
        return None

    if not np.isfinite(pcm).all():
        log.warning("[NC] pcm contains NaN/Inf")
        return None

    return pcm



def _pick_noise_reference(pcm: np.ndarray, sample_rate: int) -> np.ndarray:

    n = max(1, int(NOISE_SAMPLE_SECS * sample_rate))

    if len(pcm) <= n:
        return pcm

    head = pcm[:n]
    tail = pcm[-n:]

    return head if _rms(head) <= _rms(tail) else tail



def _denoise_spectral_fallback(pcm: np.ndarray) -> np.ndarray:
    try:
        frame = NOISE_FRAME_SIZE
        hop = NOISE_HOP_SIZE

        if len(pcm) < frame:
            return pcm

        win = np.hanning(frame).astype(np.float32)

        frames = [pcm[i:i + frame] * win for i in range(0, len(pcm) - frame + 1, hop)]
        if not frames:
            return pcm

        spectra = np.array([np.fft.rfft(f) for f in frames])
        mags = np.abs(spectra)

        # quiet frames selection
        quiet_count = max(1, int(len(mags) * NOISE_QUIET_FRAME_RATIO))
        quiet_idx = np.argsort(mags.mean(axis=1))[:quiet_count]
        noise_profile = mags[quiet_idx].mean(axis=0)

        alpha = NOISE_ALPHA_BASE + (NOISE_CANCEL_STRENGTH * NOISE_ALPHA_SCALE)

        clean_mag = np.maximum(mags - alpha * noise_profile, 0.0)
        clean_spec = clean_mag * np.exp(1j * np.angle(spectra))

        out = np.zeros(len(pcm), dtype=np.float32)
        norm = np.zeros(len(pcm), dtype=np.float32)

        for idx, start in enumerate(range(0, len(pcm) - frame + 1, hop)):
            frm = np.fft.irfft(clean_spec[idx], n=frame).astype(np.float32)
            out[start:start + frame] += frm * win
            norm[start:start + frame] += win ** 2

        valid = norm > NOISE_NORM_EPS
        out[valid] /= norm[valid]

        mixed = (NOISE_CANCEL_STRENGTH * out) + ((1.0 - NOISE_CANCEL_STRENGTH) * pcm)
        return mixed.astype(np.float32)

    except Exception as e:
        log.warning(f"[NC] spectral fallback failed: {e}")
        return pcm



def _denoise_noisereduce(pcm: np.ndarray, sample_rate: int) -> np.ndarray:

    nr = _get_noisereduce()
    if nr is None:
        return _denoise_spectral_fallback(pcm)

    try:
        noise_ref = _pick_noise_reference(pcm, sample_rate)

        denoised = nr.reduce_noise(
            y=pcm,
            sr=sample_rate,
            y_noise=noise_ref,
            prop_decrease=NOISE_CANCEL_STRENGTH,
            stationary=True,
            use_torch=False,
        )
        return denoised.astype(np.float32)

    except Exception as e:
        log.warning(f"[NC] noisereduce failed, using fallback: {e}")
        return _denoise_spectral_fallback(pcm)



def denoise_chunk(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:

    if not NOISE_CANCEL_ENABLED:
        return pcm

    pcm = _safe_float32_pcm(pcm)
    if pcm is None or len(pcm) == 0:
        return pcm if pcm is not None else np.array([], dtype=np.float32)

    try:
        out = _denoise_noisereduce(pcm, sample_rate)
        _stats["processed"] += 1
        log.debug(f"[NC] rms {_rms(pcm):.5f} -> {_rms(out):.5f}")
        return out

    except Exception as e:
        log.error(f"[NC] denoise_chunk error: {e}")
        return pcm



def is_noise_only(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bool:

    if not NOISE_CANCEL_ENABLED:
        return False

    if pcm is None or len(pcm) == 0:
        return True

    try:
        energy = _rms(pcm)
        if energy < NOISE_RMS_FLOOR:
            _stats["dropped"] += 1
            log.debug(f"[NC] dropped chunk rms={energy:.5f} floor={NOISE_RMS_FLOOR:.5f}")
            return True
        return False

    except Exception as e:
        log.error(f"[NC] is_noise_only error: {e}")
        return False



def get_noise_stats() -> dict:

    processed = _stats["processed"]
    dropped = _stats["dropped"]

    return {
        "processed": processed,
        "dropped": dropped,
        "noise_ratio": round(dropped / processed, 3) if processed > 0 else 0.0,
        "enabled": NOISE_CANCEL_ENABLED,
        "backend": "noisereduce" if _get_noisereduce() is not None else "spectral_fallback",
        "strength": NOISE_CANCEL_STRENGTH,
        "rms_floor": NOISE_RMS_FLOOR,
        "noise_sample_secs": NOISE_SAMPLE_SECS,
    }



def reset_stats() -> None:
    _stats["processed"] = 0
    _stats["dropped"] = 0

