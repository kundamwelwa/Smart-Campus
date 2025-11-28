"""Academic Service Main Application"""
import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from services.academic_service.api import (
    admin,
    assignments,
    courses,
    enrollments,
    grades,
    lecturer,
    sections,
)
from shared.config import settings
from shared.database import init_db, init_mongodb

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting Academic Service")

    # Retry database connection with backoff
    for attempt in range(5):
        try:
            await init_db()
            await init_mongodb()
            logger.info("Academic Service ready - databases connected")
            break
        except Exception as e:
            if attempt < 4:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8 seconds
                logger.warning(f"Database connection failed (attempt {attempt + 1}/5), retrying in {wait_time}s", error=str(e))
                await asyncio.sleep(wait_time)
            else:
                logger.error("Failed to connect to databases after 5 attempts - starting anyway")

    yield
    logger.info("Academic Service shutdown complete")


app = FastAPI(
    title="Argos Academic Service",
    description="Course and enrollment management service with policy engine and event sourcing",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Include routers
app.include_router(courses.router, prefix="/api/v1/courses", tags=["Courses"])
app.include_router(sections.router, prefix="/api/v1/sections", tags=["Sections"])
app.include_router(enrollments.router, prefix="/api/v1/enrollments", tags=["Enrollments"])
app.include_router(grades.router, prefix="/api/v1/academic", tags=["Grades"])
app.include_router(assignments.router, prefix="/api/v1", tags=["Assignments"])
app.include_router(lecturer.router, prefix="/api/v1", tags=["Lecturer"])
app.include_router(admin.router, prefix="/api/v1", tags=["Admin"])


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "academic_service",
    }


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "service": "Academic Service",
        "version": "0.1.0",
        "status": "operational",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.academic_service.main:app",
        host="0.0.0.0",
        port=settings.academic_service_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )

