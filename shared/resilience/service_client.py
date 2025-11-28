"""
Service Client with Circuit Breaker and Graceful Degradation

Provides resilient HTTP client for calling external services with:
- Circuit breaker pattern
- Timeout handling
- Graceful degradation
"""

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
import structlog

from shared.domain.exceptions import (
    CircuitBreakerOpenError,
    ExternalServiceError,
)
from shared.resilience.circuit_breaker import CircuitBreakerConfig, circuit_breaker_manager

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class ResilientServiceClient:
    """
    HTTP client with circuit breaker and graceful degradation.

    Automatically handles:
    - Circuit breaker pattern
    - Timeouts
    - Retries (optional)
    - Graceful degradation with fallback functions
    """

    def __init__(
        self,
        service_name: str,
        base_url: str,
        timeout: float = 10.0,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
    ):
        """
        Initialize resilient service client.

        Args:
            service_name: Name of the service (for circuit breaker)
            base_url: Base URL of the service
            timeout: Request timeout in seconds
            circuit_breaker_config: Optional circuit breaker configuration
        """
        self.service_name = service_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.circuit_breaker = circuit_breaker_manager.get_breaker(
            service_name,
            circuit_breaker_config,
        )

    async def call(
        self,
        method: str,
        path: str,
        fallback: Callable[[], T] | None = None,
        **kwargs: Any
    ) -> Any:
        """
        Make HTTP request with circuit breaker and fallback.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: URL path (will be appended to base_url)
            fallback: Optional fallback function if service fails
            **kwargs: Additional arguments for httpx request

        Returns:
            Response data or fallback result

        Raises:
            ExternalServiceError: If service fails and no fallback
            CircuitBreakerOpenError: If circuit breaker is open
        """
        url = f"{self.base_url}{path}"

        async def _make_request():
            """Make the actual HTTP request."""
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json() if response.content else None

        try:
            # Call through circuit breaker
            return await self.circuit_breaker.call(_make_request)

        except CircuitBreakerOpenError:
            # Circuit breaker is open
            logger.warning(
                "Circuit breaker open, using fallback",
                service=self.service_name,
                path=path,
            )
            if fallback:
                return await self._execute_fallback(fallback)
            raise

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            # Timeout or connection error
            logger.warning(
                "Service request failed (timeout/connection)",
                service=self.service_name,
                path=path,
                error=str(e),
            )
            if fallback:
                return await self._execute_fallback(fallback)
            raise ExternalServiceError(
                service_name=self.service_name,
                message=f"Request to {path} failed: {str(e)}",
                timeout=isinstance(e, httpx.TimeoutException),
                cause=e,
            )

        except httpx.HTTPStatusError as e:
            # HTTP error status
            logger.warning(
                "Service returned error status",
                service=self.service_name,
                path=path,
                status_code=e.response.status_code,
            )
            if fallback:
                return await self._execute_fallback(fallback)
            raise ExternalServiceError(
                service_name=self.service_name,
                message=f"Service returned {e.response.status_code}",
                cause=e,
            )

        except Exception as e:
            # Other errors
            logger.error(
                "Unexpected error calling service",
                service=self.service_name,
                path=path,
                error=str(e),
                error_type=type(e).__name__,
            )
            if fallback:
                return await self._execute_fallback(fallback)
            raise ExternalServiceError(
                service_name=self.service_name,
                message=f"Unexpected error: {str(e)}",
                cause=e,
            )

    async def _execute_fallback(self, fallback: Callable[[], T]) -> T:
        """
        Execute fallback function.

        Args:
            fallback: Fallback function

        Returns:
            Result from fallback
        """
        logger.info(
            "Executing fallback function",
            service=self.service_name,
        )

        try:
            if asyncio.iscoroutinefunction(fallback):
                result = await fallback()
            else:
                result = fallback()

            logger.info(
                "Fallback executed successfully",
                service=self.service_name,
            )
            return result

        except Exception as e:
            logger.error(
                "Fallback function failed",
                service=self.service_name,
                error=str(e),
            )
            raise

    async def get(self, path: str, fallback: Callable[[], T] | None = None, **kwargs) -> Any:
        """Make GET request."""
        return await self.call("GET", path, fallback=fallback, **kwargs)

    async def post(self, path: str, fallback: Callable[[], T] | None = None, **kwargs) -> Any:
        """Make POST request."""
        return await self.call("POST", path, fallback=fallback, **kwargs)

    async def put(self, path: str, fallback: Callable[[], T] | None = None, **kwargs) -> Any:
        """Make PUT request."""
        return await self.call("PUT", path, fallback=fallback, **kwargs)

    async def delete(self, path: str, fallback: Callable[[], T] | None = None, **kwargs) -> Any:
        """Make DELETE request."""
        return await self.call("DELETE", path, fallback=fallback, **kwargs)


# ML Service client with rule-based fallback
class MLServiceClient(ResilientServiceClient):
    """
    Specialized client for ML service with rule-based fallback.

    Automatically falls back to rule-based predictions when ML service is unavailable.
    """

    async def predict_enrollment(
        self,
        student_data: dict[str, Any],
        rule_based_fallback: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Predict enrollment with automatic fallback to rule-based logic.

        Args:
            student_data: Student data for prediction
            rule_based_fallback: Function that implements rule-based prediction

        Returns:
            Prediction result (from ML or rule-based)
        """
        async def _ml_fallback():
            """Fallback to rule-based prediction."""
            logger.info(
                "ML service unavailable, using rule-based fallback",
                service=self.service_name,
            )
            result = rule_based_fallback(student_data)
            result["fallback_used"] = True
            result["confidence"] = result.get("confidence", 0.6)  # Lower confidence for fallback
            return result

        try:
            result = await self.post(
                "/api/v1/predict/enrollment",
                json=student_data,
                fallback=_ml_fallback,
            )
            result["fallback_used"] = False
            return result

        except (ExternalServiceError, CircuitBreakerOpenError) as e:
            # Use fallback
            logger.warning(
                "ML service error, using rule-based fallback",
                service=self.service_name,
                error=str(e),
            )
            return await _ml_fallback()

