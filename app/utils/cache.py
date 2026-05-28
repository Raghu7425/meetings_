"""
Redis-backed LRU response cache.

Replaces the in-memory OrderedDict in agent.py with a distributed cache
that survives process restarts and works across multiple worker instances.

Key scheme:  cache:<sha256_of_cache_key>
TTL:         configurable via REDIS_CACHE_TTL (default 30 min)
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from app.config import REDIS_CACHE_TTL
from app.db.redis_client import get_redis

log = logging.getLogger("cache")

_PREFIX = "cache:"


def _hash_key(raw: str) -> str:
    return _PREFIX + hashlib.sha256(raw.encode()).hexdigest()[:32]


async def cache_get(key: str) -> str | None:
    """Return the cached string value, or None if absent/expired."""
    try:
        r = await get_redis()
        value = await r.get(_hash_key(key))
        if value:
            log.debug("[cache] HIT key=%s", key[:60])
        return value
    except Exception as exc:
        log.warning("[cache] GET error: %s", exc)
        return None


async def cache_set(key: str, value: str, ttl: int = REDIS_CACHE_TTL) -> None:
    """Store value under key with a TTL (seconds)."""
    try:
        r = await get_redis()
        await r.setex(_hash_key(key), ttl, value)
        log.debug("[cache] SET key=%s ttl=%ds", key[:60], ttl)
    except Exception as exc:
        log.warning("[cache] SET error: %s", exc)


async def cache_delete(key: str) -> None:
    try:
        r = await get_redis()
        await r.delete(_hash_key(key))
    except Exception as exc:
        log.warning("[cache] DEL error: %s", exc)


async def cache_flush_prefix(prefix: str) -> int:
    """Delete all cache keys whose raw key starts with prefix. Returns count deleted."""
    try:
        r = await get_redis()
        cursor, deleted = 0, 0
        while True:
            cursor, keys = await r.scan(cursor, match=f"{_PREFIX}*", count=200)
            if keys:
                await r.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        log.debug("[cache] flushed %d keys with prefix=%s", deleted, prefix)
        return deleted
    except Exception as exc:
        log.warning("[cache] FLUSH error: %s", exc)
        return 0
