"""
API Gateway Middleware

Custom middleware for logging, rate limiting, and request tracking.
"""

import time
from collections.abc import Callable
from uuid import uuid4

import structlog
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.responses import Response as FastAPIResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class CORSPreflightMiddleware(BaseHTTPMiddleware):
    """
    Middleware to explicitly handle OPTIONS preflight requests.

    This ensures CORS preflight requests are handled before
    they reach other middleware or route handlers.
    """

    def __init__(self, app: Callable, allowed_origins: list[str]):
        """
        Initialize CORS preflight middleware.

        Args:
            app: ASGI app
            allowed_origins: List of allowed CORS origins
        """
        super().__init__(app)
        self.allowed_origins = allowed_origins

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Handle OPTIONS preflight requests.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response: Preflight response or passes through
        """
        origin = request.headers.get("origin")

        # In development, be more permissive - allow localhost origins
        # Check if origin matches any allowed origin or is a localhost variant
        is_allowed_origin = False
        if origin:
            # Direct match
            if origin in self.allowed_origins or any(
                origin.startswith(("http://localhost", "http://127.0.0.1"))
                for allowed in self.allowed_origins
                if allowed.startswith(("http://localhost", "http://127.0.0.1"))
            ) or "*" in self.allowed_origins:
                is_allowed_origin = True
        # If no origins specified, allow all (dev mode)
        elif not self.allowed_origins:
            is_allowed_origin = True


        if request.method == "OPTIONS":
            response = FastAPIResponse(status_code=200)
            # Always set the origin header if present (permissive approach for development)
            if origin:
                response.headers["Access-Control-Allow-Origin"] = origin
            else:
                response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept, Origin"
            response.headers["Access-Control-Max-Age"] = "3600"
            logger.info(
                "Handled OPTIONS preflight",
                origin=origin,
                path=request.url.path,
                allowed=is_allowed_origin,
                headers=dict(response.headers)
            )
            return response

        # For non-OPTIONS requests, continue to next middleware
        response = await call_next(request)

        # Add CORS headers to all responses (as fallback)
        if origin:
            if is_allowed_origin:
                response.headers["Access-Control-Allow-Origin"] = origin
            else:
                # Still add header for development (permissive)
                response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept, Origin"

        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all requests and responses.

    Adds correlation IDs and tracks request duration.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with logging.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response: HTTP response
        """
        # Generate correlation ID
        correlation_id = str(uuid4())
        request.state.correlation_id = correlation_id

        # Log request
        start_time = time.time()

        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            correlation_id=correlation_id,
            client_host=request.client.host if request.client else None,
        )

        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log response
            logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
                correlation_id=correlation_id,
            )

            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id

            return response

        except Exception as e:
            duration = time.time() - start_time

            logger.error(
                "Request failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                duration_ms=round(duration * 1000, 2),
                correlation_id=correlation_id,
            )
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware.

    Limits requests per IP address using in-memory counter.
    For production, use Redis-based rate limiting.
    """

    def __init__(self, app: Callable, requests_per_minute: int = 100):
        """
        Initialize rate limiter.

        Args:
            app: ASGI app
            requests_per_minute: Maximum requests per minute per IP
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with rate limiting.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response: HTTP response or rate limit error
        """
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Skip rate limiting for health checks and OPTIONS (preflight) requests
        if request.url.path.startswith("/health") or request.method == "OPTIONS":
            return await call_next(request)

        # Current time
        now = time.time()
        minute_ago = now - 60

        # Clean old entries and count recent requests
        if client_ip in self.request_counts:
            self.request_counts[client_ip] = [
                t for t in self.request_counts[client_ip] if t > minute_ago
            ]
        else:
            self.request_counts[client_ip] = []

        # Check rate limit
        if len(self.request_counts[client_ip]) >= self.requests_per_minute:
            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                requests=len(self.request_counts[client_ip]),
                limit=self.requests_per_minute,
            )

            # Create rate limit response with CORS headers
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute allowed",
                },
            )

            # Add CORS headers to rate limit response
            origin = request.headers.get("origin")
            if origin:
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "*"

            return response

        # Record request
        self.request_counts[client_ip].append(now)

        # Process request
        response = await call_next(request)

        # Ensure CORS headers are added to all responses
        origin = request.headers.get("origin")
        if origin:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"

        return response

