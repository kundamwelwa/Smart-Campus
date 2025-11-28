"""
MongoDB Connection

Async MongoDB client for event store and document storage.
"""


import structlog
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from shared.config import settings

logger = structlog.get_logger(__name__)


# Global MongoDB client
_mongodb_client: AsyncIOMotorClient | None = None


async def init_mongodb() -> AsyncIOMotorClient:
    """
    Initialize MongoDB connection.

    Returns:
        AsyncIOMotorClient: MongoDB client instance
    """
    global _mongodb_client

    if _mongodb_client is not None:
        return _mongodb_client

    _mongodb_client = AsyncIOMotorClient(settings.mongodb_url)

    # Verify connection
    try:
        await _mongodb_client.admin.command("ping")
        logger.info("MongoDB connection established", host=settings.mongodb_host)
    except Exception as e:
        logger.error("Failed to connect to MongoDB", error=str(e))
        raise

    return _mongodb_client


async def get_mongodb() -> AsyncIOMotorDatabase:
    """
    Get MongoDB database instance.

    Returns:
        AsyncIOMotorDatabase: Database instance
    """
    if _mongodb_client is None:
        await init_mongodb()

    return _mongodb_client[settings.mongodb_db]  # type: ignore


async def close_mongodb() -> None:
    """Close MongoDB connections."""
    global _mongodb_client

    if _mongodb_client is not None:
        _mongodb_client.close()
        _mongodb_client = None
        logger.info("MongoDB connections closed")

