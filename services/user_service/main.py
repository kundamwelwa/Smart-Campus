"""
User Service Main Application

Microservice handling user management, authentication, and authorization.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from services.user_service.api import admin, auth, roles, users
from shared.config import settings
from shared.database import close_db, close_mongodb, close_redis, init_db, init_mongodb, init_redis

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting User Service")

    # Retry database connection with backoff
    for attempt in range(5):
        try:
            await init_db()
            await init_mongodb()
            await init_redis()
            logger.info("User Service ready - databases connected")
            break
        except Exception as e:
            if attempt < 4:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8 seconds
                logger.warning(f"Database connection failed (attempt {attempt + 1}/5), retrying in {wait_time}s", error=str(e))
                await asyncio.sleep(wait_time)
            else:
                logger.error("Failed to connect to databases after 5 attempts - starting anyway")

    yield

    await close_db()
    await close_mongodb()
    await close_redis()
    logger.info("User Service shutdown complete")


app = FastAPI(
    title="Argos User Service",
    description="User management, authentication, and authorization service",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(roles.router, prefix="/api/v1/roles", tags=["Roles"])
app.include_router(admin.router, prefix="/api/v1", tags=["Admin"])


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "user_service"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.user_service.main:app",
        host="0.0.0.0",
        port=settings.user_service_port,
        reload=settings.debug,
    )

