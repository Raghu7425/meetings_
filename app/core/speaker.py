"""
Handles speaker embedding and speaker identification.

- Loads speaker recognition model (SpeechBrain)
- Selects device (CPU/GPU) for processing

- Cleans audio:
  • removes silence
  • normalizes audio values

- Converts audio into embeddings (voice representation)
- Compares embeddings using similarity score
- Classifies if current speaker matches primary user

- Builds strong primary speaker profile from multiple samples
- Updates and averages embeddings for better accuracy

This file identifies and verifies the user's voice using audio data.
"""


import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from speechbrain.inference.speaker import EncoderClassifier
from app.config import (SPEAKER_THRESHOLD, SPEECH_SILENCE_THRESHOLD, 
                        SPEECH_SAMPLE_RATE, SPEAKER_ENROLLMENT_SAMPLES)


log = logging.getLogger("speaker")
_classifier = None


def get_device() -> str:

    forced = os.environ.get("SPEAKER_DEVICE", "").strip().lower()

    if forced in {"cpu", "cuda"}:
        if forced == "cuda" and not torch.cuda.is_available():
            log.warning("[speaker] SPEAKER_DEVICE=cuda but CUDA is not available. Falling back to cpu.")
            return "cpu"
        return forced

    return "cuda" if torch.cuda.is_available() else "cpu"



def get_classifier():

    global _classifier

    if _classifier is None:
        device = get_device()
        log.info(f"[speaker] Loading SpeechBrain ECAPA-TDNN on {device}...")
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )
        log.info("[speaker] SpeechBrain speaker model ready.")

    return _classifier



def trim_silence(audio_np: np.ndarray, threshold: float = SPEECH_SILENCE_THRESHOLD) -> np.ndarray | None:

    if audio_np is None or len(audio_np) == 0:
        return None

    try:
        audio = np.asarray(audio_np, dtype=np.float32).flatten()
        if audio.size == 0:
            return None

        idx = np.where(np.abs(audio) > threshold)[0]
        if len(idx) == 0:
            return audio

        return audio[idx[0]: idx[-1] + 1]

    except Exception as e:
        log.error(f"[speaker] silence trim failed: {e}")
        return None



def _normalize_audio(audio_np: np.ndarray) -> np.ndarray | None:

    if audio_np is None or len(audio_np) == 0:
        return None

    try:
        wav = np.asarray(audio_np, dtype=np.float32).flatten()
        if wav.size == 0 or not np.isfinite(wav).all():
            return None

        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav / max(peak, 1e-6)

        return wav

    except Exception as e:
        log.error(f"[speaker] audio normalization failed: {e}")
        return None



def _prepare_waveform(audio_np: np.ndarray, sample_rate: int = SPEECH_SAMPLE_RATE) -> torch.Tensor | None:

    wav = _normalize_audio(audio_np)
    if wav is None:
        return None

    try:
        if sample_rate != 16000:
            log.warning(f"[speaker] Expected 16kHz audio, got {sample_rate}Hz. Speaker accuracy may drop.")

        return torch.from_numpy(wav).float().unsqueeze(0)  # [1, T]

    except Exception as e:
        log.error(f"[speaker] waveform prep failed: {e}")
        return None


def extract_embedding(audio_np: np.ndarray, sample_rate: int = SPEECH_SAMPLE_RATE, trim: bool = True):

    try:
        audio = audio_np

        if trim:
            audio = trim_silence(audio, threshold=SPEECH_SILENCE_THRESHOLD)

        if audio is None or len(audio) == 0:
            return None

        wav = _prepare_waveform(audio, sample_rate)
        if wav is None:
            return None

        classifier = get_classifier()
        param_device = next(classifier.mods.parameters()).device
        wav = wav.to(param_device)

        with torch.no_grad():
            emb = classifier.encode_batch(wav)

        emb = emb.squeeze()
        if emb.ndim != 1:
            emb = emb.reshape(-1)

        emb = F.normalize(emb, p=2, dim=0)
        return emb.detach().cpu().numpy().astype(np.float32)

    except Exception as e:
        log.error(f"[speaker] embedding extraction failed: {e}")
        return None



def cosine_similarity(a, b) -> float:

    if a is None or b is None:
        return 0.0

    try:
        return float(1 - cosine(a, b))
    
    except Exception as e:
        log.error(f"[speaker] similarity failed: {e}")
        return 0.0



def classify_speaker(current_emb, primary_emb, threshold=SPEAKER_THRESHOLD):

    if primary_emb is None or current_emb is None:
        return False, 0.0

    score = cosine_similarity(primary_emb, current_emb)

    return score >= threshold, score        # TRUE/FALSE, SIMILARITY SCORE



def average_embeddings(embeddings):

    if not embeddings:
        return None

    try:
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None

        mean_emb = np.mean(arr, axis=0)
        norm = np.linalg.norm(mean_emb)
        if not np.isfinite(norm) or norm < 1e-8:
            return None

        return (mean_emb / norm).astype(np.float32)

    except Exception as e:
        log.error(f"[speaker] average_embeddings failed: {e}")
        return None



def build_primary_embedding(embeddings):
    """
    Build a robust primary embedding from the first 3 enrollment embeddings.

    Strategy:
    - If fewer than 2 embeddings -> fallback to average_embeddings
    - If exactly 2 embeddings -> average both
    - If 3 embeddings:
        1. compute pairwise similarities
        2. find the best matching pair
        3. if the third embedding is also reasonably close to that pair,
           average all 3
        4. otherwise average only the best pair
    """
    if not embeddings:
        return None

    try:
        if len(embeddings) < SPEAKER_ENROLLMENT_SAMPLES:
            return average_embeddings(embeddings)

        emb1, emb2, emb3 = embeddings[:3]

        sim12 = cosine_similarity(emb1, emb2)
        sim13 = cosine_similarity(emb1, emb3)
        sim23 = cosine_similarity(emb2, emb3)

        pairs = [
            ((0, 1), sim12),
            ((0, 2), sim13),
            ((1, 2), sim23),
        ]

        # best pair = highest similarity
        best_pair, best_score = max(pairs, key=lambda x: x[1])
        i, j = best_pair

        # the remaining third index
        third_idx = ({0, 1, 2} - {i, j}).pop()

        # average of best pair
        pair_avg = average_embeddings([embeddings[i], embeddings[j]])
        if pair_avg is None:
            return None

        # compare third embedding with best-pair average
        third_score = cosine_similarity(pair_avg, embeddings[third_idx])

        log.info(
            f"[speaker] primary build sims: "
            f"sim12={sim12:.3f}, sim13={sim13:.3f}, sim23={sim23:.3f}, "
            f"best_pair={best_pair}, best_score={best_score:.3f}, "
            f"third_score={third_score:.3f}"
        )

        # if third embedding is also reasonably close, use all 3 else use only best pair
        if third_score >= SPEAKER_THRESHOLD:
            return average_embeddings([emb1, emb2, emb3])

        return pair_avg

    except Exception as e:
        log.error(f"[speaker] build_primary_embedding failed: {e}")
        return None
    
