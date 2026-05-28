"""
Tenacity retry decorators for LLM and external service calls.

Provides pre-configured decorators:
  @llm_retry       — exponential back-off for Ollama calls (HTTPError, timeout)
  @storage_retry   — lighter retry for MinIO / Redis I/O
  @idempotent_retry — generic retry with jitter for any idempotent operation

All decorators log each attempt so failures are visible in the structured log.
"""

from __future__ import annotations

import logging
from typing import Callable, TypeVar, Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
    wait_random_exponential,
    before_sleep_log,
    after_log,
)

from app.config import (
    LLM_RETRY_MAX_ATTEMPTS,
    LLM_RETRY_MIN_WAIT,
    LLM_RETRY_MAX_WAIT,
)

log = logging.getLogger("retry")

_RETRIABLE_HTTP = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    ConnectionError,
)


def llm_retry(func: Callable | None = None):
    """
    Decorator: retry an async LLM call up to LLM_RETRY_MAX_ATTEMPTS times.
    Uses randomised exponential back-off to avoid thundering herd on Ollama.
    """
    decorator = retry(
        reraise=True,
        stop=stop_after_attempt(LLM_RETRY_MAX_ATTEMPTS),
        wait=wait_random_exponential(
            min=LLM_RETRY_MIN_WAIT,
            max=LLM_RETRY_MAX_WAIT,
        ),
        retry=retry_if_exception_type((*_RETRIABLE_HTTP, json.JSONDecodeError if False else Exception)),
        before_sleep=before_sleep_log(log, logging.WARNING),
        after=after_log(log, logging.DEBUG),
    )
    if func is not None:
        return decorator(func)
    return decorator


def storage_retry(func: Callable | None = None):
    """Light retry (3 attempts, 0.5–5 s back-off) for object storage calls."""
    decorator = retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=5.0, jitter=1.0),
        retry=retry_if_exception_type((*_RETRIABLE_HTTP, OSError, IOError)),
        before_sleep=before_sleep_log(log, logging.WARNING),
    )
    if func is not None:
        return decorator(func)
    return decorator


async def retry_async(
    coro_func: Callable,
    *args: Any,
    max_attempts: int = LLM_RETRY_MAX_ATTEMPTS,
    min_wait: float = LLM_RETRY_MIN_WAIT,
    max_wait: float = LLM_RETRY_MAX_WAIT,
    **kwargs: Any,
) -> Any:
    """
    Functional-style async retry — useful when you cannot use a decorator
    (e.g. wrapping a lambda or a partial).

    Example:
        result = await retry_async(call_ollama, prompt, max_attempts=3)
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(min=min_wait, max=max_wait),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.WARNING),
    ):
        with attempt:
            return await coro_func(*args, **kwargs)


# ── import guard ───────────────────────────────────────────────────────────────
import json  # noqa: E402 — needed for json.JSONDecodeError reference above
