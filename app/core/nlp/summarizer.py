"""
Extractive summarization via Sumy — no LLM required.

Three algorithms optimized for distinct audiences:
  executive_summary() → TextRank  — narrative, accessible prose
  technical_summary() → LSA       — latent semantic key-points
  business_summary()  → Luhn      — high-importance business sentences

Speed: ~10-50 ms for a 10k-word transcript (100-300× faster than LLM).
All sync; async variants run in thread pool.

Falls back gracefully when sumy is not installed (returns truncated text).
"""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger("nlp.summarizer")

_TOKENIZER: object | None = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from sumy.nlp.tokenizers import Tokenizer
        _TOKENIZER = Tokenizer("english")
    return _TOKENIZER


def _sumy(text: str, sentences: int, method: str) -> str:
    if not text.strip():
        return ""
    try:
        from sumy.parsers.plaintext import PlaintextParser

        parser = PlaintextParser.from_string(text, _tokenizer())

        if method == "textrank":
            from sumy.summarizers.text_rank import TextRankSummarizer
            s = TextRankSummarizer()
        elif method == "luhn":
            from sumy.summarizers.luhn import LuhnSummarizer
            s = LuhnSummarizer()
        else:  # lsa
            from sumy.summarizers.lsa import LsaSummarizer
            s = LsaSummarizer()

        return " ".join(str(sent) for sent in s(parser.document, sentences))

    except ImportError:
        log.warning("sumy not installed — pip install sumy")
        # Naive fallback: first N sentences
        sents = [s.strip() for s in text.replace("? ", ". ").replace("! ", ". ").split(". ") if s.strip()]
        return ". ".join(sents[:sentences]) + ("." if sents else "")
    except Exception as exc:
        log.warning("Sumy failed (%s) — returning truncated text", exc)
        return text[:800]


# ── public sync API ───────────────────────────────────────────────────────────

def executive_summary(text: str, sentences: int = 4) -> str:
    """TextRank — readable narrative; best for non-technical stakeholders."""
    return _sumy(text, sentences, "textrank")


def technical_summary(text: str, sentences: int = 6) -> str:
    """LSA — latent semantic analysis; captures scattered technical points."""
    return _sumy(text, sentences, "lsa")


def business_summary(text: str, sentences: int = 5) -> str:
    """Luhn — frequency-weighted; highlights important business sentences."""
    return _sumy(text, sentences, "luhn")


def multi_summary(text: str) -> dict[str, str]:
    """All three flavors in one call."""
    return {
        "executive": executive_summary(text),
        "technical": technical_summary(text),
        "business":  business_summary(text),
    }


# ── async wrappers ────────────────────────────────────────────────────────────

async def executive_summary_async(text: str, sentences: int = 4) -> str:
    return await asyncio.to_thread(executive_summary, text, sentences)


async def technical_summary_async(text: str, sentences: int = 6) -> str:
    return await asyncio.to_thread(technical_summary, text, sentences)


async def business_summary_async(text: str, sentences: int = 5) -> str:
    return await asyncio.to_thread(business_summary, text, sentences)


async def multi_summary_async(text: str) -> dict[str, str]:
    return await asyncio.to_thread(multi_summary, text)
