"""
Database Connection and Utilities

Manages connections to PostgreSQL, MongoDB, and Redis.
"""

from shared.database.mongodb import close_mongodb, get_mongodb, init_mongodb
from shared.database.postgres import Base, close_db, get_db, init_db
from shared.database.redis import close_redis, get_redis, init_redis

__all__ = [
    "get_db",
    "init_db",
    "close_db",
    "Base",
    "get_mongodb",
    "init_mongodb",
    "close_mongodb",
    "get_redis",
    "init_redis",
    "close_redis",
]

