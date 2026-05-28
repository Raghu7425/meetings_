"""
Async Redis connection pool.

Provides a shared, lazily-initialised redis.asyncio connection pool used by:
  - Job state (hash:job:<id>)
  - Session state (hash:session:<id>)
  - Rolling summary cache (str:summary:<id>)
  - Redis Streams pipeline events
  - Response LRU cache (replaces in-memory OrderedDict)
"""

from __future__ import annotations

import logging
from typing import Any

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool

from app.config import REDIS_URL, REDIS_MAX_CONNECTIONS, REDIS_SOCKET_TIMEOUT

log = logging.getLogger("redis_client")

_pool: ConnectionPool | None = None
_client: aioredis.Redis | None = None


def _build_pool() -> ConnectionPool:
    return aioredis.ConnectionPool.from_url(
        REDIS_URL,
        max_connections=REDIS_MAX_CONNECTIONS,
        socket_timeout=REDIS_SOCKET_TIMEOUT,
        socket_connect_timeout=REDIS_SOCKET_TIMEOUT,
        decode_responses=True,
        health_check_interval=30,
    )


async def get_redis() -> aioredis.Redis:
    """Return the shared Redis client, creating it on first call."""
    global _pool, _client
    if _client is None:
        _pool = _build_pool()
        _client = aioredis.Redis(connection_pool=_pool)
        log.info("[redis] Connection pool initialised → %s", REDIS_URL)
    return _client


async def close_redis() -> None:
    """Gracefully close the connection pool (call on app shutdown)."""
    global _pool, _client
    if _client is not None:
        await _client.aclose()
        _client = None
    if _pool is not None:
        await _pool.aclose()
        _pool = None
    log.info("[redis] Connection pool closed")


async def ping_redis() -> bool:
    """Health check — returns True if Redis responds."""
    try:
        r = await get_redis()
        return await r.ping()
    except Exception as exc:
        log.warning("[redis] Health check failed: %s", exc)
        return False


class RedisJobStore:
    """Thin wrapper for persisting pipeline job state in Redis hashes."""

    PREFIX = "job:"

    def __init__(self, redis: aioredis.Redis, ttl: int = 86400) -> None:
        self._r = redis
        self._ttl = ttl

    def _key(self, job_id: str) -> str:
        return f"{self.PREFIX}{job_id}"

    async def set(self, job_id: str, data: dict[str, Any]) -> None:
        key = self._key(job_id)
        # Redis hashes require string values
        str_data = {k: str(v) if v is not None else "" for k, v in data.items()}
        await self._r.hset(key, mapping=str_data)
        await self._r.expire(key, self._ttl)

    async def update(self, job_id: str, **fields: Any) -> None:
        key = self._key(job_id)
        str_fields = {k: str(v) if v is not None else "" for k, v in fields.items()}
        await self._r.hset(key, mapping=str_fields)
        await self._r.expire(key, self._ttl)

    async def get(self, job_id: str) -> dict[str, str] | None:
        data = await self._r.hgetall(self._key(job_id))
        return data or None

    async def delete(self, job_id: str) -> None:
        await self._r.delete(self._key(job_id))

    async def exists(self, job_id: str) -> bool:
        return bool(await self._r.exists(self._key(job_id)))


class RedisSessionStore:
    """Thin wrapper for distributed WebSocket session state."""

    PREFIX = "session:"

    def __init__(self, redis: aioredis.Redis, ttl: int = 3600) -> None:
        self._r = redis
        self._ttl = ttl

    def _key(self, session_id: str) -> str:
        return f"{self.PREFIX}{session_id}"

    async def get_or_create(self, session_id: str) -> dict[str, str]:
        key = self._key(session_id)
        data = await self._r.hgetall(key)
        if not data:
            defaults = {
                "intent": "information_request",
                "emotion": "neutral",
                "turns": "0",
                "voice": "",
            }
            await self._r.hset(key, mapping=defaults)
            await self._r.expire(key, self._ttl)
            return defaults
        await self._r.expire(key, self._ttl)  # refresh TTL on access
        return data

    async def update(self, session_id: str, **fields: Any) -> None:
        key = self._key(session_id)
        str_fields = {k: str(v) for k, v in fields.items()}
        await self._r.hset(key, mapping=str_fields)
        await self._r.expire(key, self._ttl)

    async def delete(self, session_id: str) -> None:
        await self._r.delete(self._key(session_id))
