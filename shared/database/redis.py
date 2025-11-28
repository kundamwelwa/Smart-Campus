"""
Redis Connection

Async Redis client for caching and session storage.
"""


import redis.asyncio as aioredis
import structlog

from shared.config import settings

logger = structlog.get_logger(__name__)


# Global Redis client
_redis_client: aioredis.Redis | None = None


async def init_redis() -> aioredis.Redis:
    """
    Initialize Redis connection.

    Returns:
        aioredis.Redis: Redis client instance
    """
    global _redis_client

    if _redis_client is not None:
        return _redis_client

    _redis_client = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=50,
    )

    # Verify connection
    try:
        await _redis_client.ping()
        logger.info("Redis connection established", host=settings.redis_host)
    except Exception as e:
        logger.error("Failed to connect to Redis", error=str(e))
        raise

    return _redis_client


async def get_redis() -> aioredis.Redis:
    """
    Get Redis client instance.

    Returns:
        aioredis.Redis: Redis client
    """
    if _redis_client is None:
        await init_redis()

    return _redis_client  # type: ignore


async def close_redis() -> None:
    """Close Redis connections."""
    global _redis_client

    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None
        logger.info("Redis connections closed")

