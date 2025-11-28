"""
API Gateway Main Application

FastAPI-based API gateway providing unified access to all microservices.
"""

import sys
from pathlib import Path

# Add project root to Python path if running directly
# This allows absolute imports to work when running as a script
project_root = Path(__file__).parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.api_gateway.middleware import (
    CORSPreflightMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
)
from services.api_gateway.routers import (
    academic,
    admin,
    analytics,
    auth,
    facilities,
    facility,
    health,
    lecturer,
    scheduler,
    users,
)
from shared.config import settings
from shared.database import close_db, close_mongodb, close_redis, init_db, init_mongodb, init_redis

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events for database connections
    and resource initialization.
    """
    # Startup
    logger.info("Starting API Gateway", version=app.version)

    try:
        # Initialize databases
        await init_db()
        await init_mongodb()
        await init_redis()

        logger.info("All database connections initialized")

        yield

    finally:
        # Shutdown
        logger.info("Shutting down API Gateway")
        await close_db()
        await close_mongodb()
        await close_redis()
        logger.info("All database connections closed")


# Create FastAPI application
app = FastAPI(
    title="Argos API Gateway",
    description="Unified API for Argos Smart Campus Orchestration Platform",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

if settings.environment == "development":
    cors_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    # Also allow any localhost origin in development
    logger.info("CORS configured for development - allowing localhost origins")
else:
    # Production: use configured origins
    cors_origins = settings.cors_origins if settings.cors_origins else []

logger.info("CORS configured", origins=cors_origins, environment=settings.environment)


# Add custom middleware FIRST (executes LAST in the chain)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=100)

# Add FastAPI CORS middleware (executes in middle, adds headers to all responses)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Accept", "Origin"],
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Add CORS preflight middleware LAST (executes FIRST, handles OPTIONS immediately)
# This ensures OPTIONS requests are handled before any other processing
app.add_middleware(CORSPreflightMiddleware, allowed_origins=cors_origins)


# Import domain exceptions for proper handling
from fastapi.exceptions import HTTPException as FastAPIHTTPException

from shared.domain.exceptions import DomainException


# Global exception handler with rich domain exception support
@app.exception_handler(DomainException)
async def domain_exception_handler(request: Request, exc: DomainException) -> JSONResponse:
    """
    Handle domain exceptions with structured error responses.

    Args:
        request: FastAPI request
        exc: Domain exception

    Returns:
        JSONResponse: Structured error response
    """
    logger.warning(
        "Domain exception",
        path=request.url.path,
        method=request.method,
        error_code=exc.error_code.value,
        message=exc.message,
        status_code=exc.status_code,
        context=exc.context,
    )

    response = JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )

    # Add CORS headers
    origin = request.headers.get("origin")
    if origin and (origin in cors_origins or "*" in cors_origins or settings.environment == "development"):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"

    return response


@app.exception_handler(FastAPIHTTPException)
async def http_exception_handler(request: Request, exc: FastAPIHTTPException) -> JSONResponse:
    """
    Handle FastAPI HTTP exceptions.

    Args:
        request: FastAPI request
        exc: HTTP exception

    Returns:
        JSONResponse: Error response
    """
    logger.warning(
        "HTTP exception",
        path=request.url.path,
        method=request.method,
        status_code=exc.status_code,
        detail=exc.detail,
    )

    content = {
        "error": "HTTP_ERROR",
        "message": exc.detail if isinstance(exc.detail, str) else str(exc.detail),
        "status_code": exc.status_code,
    }

    response = JSONResponse(
        status_code=exc.status_code,
        content=content,
    )

    # Add CORS headers
    origin = request.headers.get("origin")
    if origin and (origin in cors_origins or "*" in cors_origins or settings.environment == "development"):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"

    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for unhandled errors.
    Ensures CORS headers are included in error responses.

    Args:
        request: FastAPI request
        exc: Raised exception

    Returns:
        JSONResponse: Error response with CORS headers
    """
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        error=str(exc),
        error_type=type(exc).__name__,
        exc_info=True,  # Include full traceback
    )

    # Create response with CORS headers
    response = JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": str(exc) if settings.debug else "An unexpected error occurred",
            "type": type(exc).__name__,
        },
    )

    # Add CORS headers manually to ensure they're present
    origin = request.headers.get("origin")
    if origin and (origin in cors_origins or "*" in cors_origins or settings.environment == "development"):
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"

    return response


# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(auth.router, prefix=f"{settings.api_v1_prefix}/auth", tags=["Authentication"])
app.include_router(users.router, prefix=f"{settings.api_v1_prefix}/users", tags=["Users"])
app.include_router(
    academic.router, prefix=f"{settings.api_v1_prefix}/academic", tags=["Academic"]
)
# Direct courses route for /api/v1/courses (also available at /api/v1/academic/courses)
app.include_router(
    academic.courses_router, prefix=f"{settings.api_v1_prefix}/courses", tags=["Courses"]
)
app.include_router(
    scheduler.router, prefix=f"{settings.api_v1_prefix}/scheduler", tags=["Scheduler"]
)
app.include_router(
    analytics.router, prefix=f"{settings.api_v1_prefix}/analytics", tags=["Analytics"]
)
app.include_router(
    facility.router, prefix=f"{settings.api_v1_prefix}/facilities", tags=["Facilities"]
)
app.include_router(
    lecturer.router, prefix=f"{settings.api_v1_prefix}", tags=["Lecturer"]
)
app.include_router(
    admin.router, prefix=f"{settings.api_v1_prefix}", tags=["Admin"]
)
app.include_router(
    facilities.router, prefix=f"{settings.api_v1_prefix}", tags=["Facilities-Public"]
)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "service": "Argos API Gateway",
        "version": "0.1.0",
        "status": "operational",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.api_gateway.main:app",
        host=settings.api_gateway_host,
        port=settings.api_gateway_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )

