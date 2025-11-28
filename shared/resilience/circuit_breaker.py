"""
Circuit Breaker Pattern Implementation

Provides fault tolerance for external service calls by preventing
cascading failures and allowing services to recover.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Number of failures before opening
    success_threshold: int = 2  # Number of successes in half-open to close
    timeout_seconds: int = 60  # Time before attempting half-open
    timeout_duration: timedelta = field(init=False)

    def __post_init__(self):
        """Initialize timeout duration."""
        self.timeout_duration = timedelta(seconds=self.timeout_seconds)


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    opened_at: datetime | None = None


class CircuitBreaker:
    """
    Circuit breaker for external service calls.

    Implements the circuit breaker pattern to prevent cascading failures:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject requests immediately
    - HALF_OPEN: Testing if service recovered, allow limited requests
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name of the circuit breaker (usually service name)
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Execute function call with circuit breaker protection.

        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from function call

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception from function call
        """
        async with self._lock:
            # Check if circuit is open
            if self.stats.state == CircuitState.OPEN:
                # Check if timeout has passed
                if self.stats.last_failure_time:
                    elapsed = datetime.utcnow() - self.stats.last_failure_time
                    if elapsed >= self.config.timeout_duration:
                        # Transition to half-open
                        logger.info(
                            "Circuit breaker transitioning to half-open",
                            circuit_name=self.name,
                            elapsed_seconds=elapsed.total_seconds(),
                        )
                        self.stats.state = CircuitState.HALF_OPEN
                        self.stats.success_count = 0
                    else:
                        # Still in timeout period
                        from shared.domain.exceptions import CircuitBreakerOpenError
                        raise CircuitBreakerOpenError(service_name=self.name)
                else:
                    # No failure time recorded, should not happen
                    from shared.domain.exceptions import CircuitBreakerOpenError
                    raise CircuitBreakerOpenError(service_name=self.name)

            # Circuit is closed or half-open, attempt call
            self.stats.total_requests += 1

        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success
            await self._record_success()
            return result

        except Exception:
            # Failure
            await self._record_failure()
            raise

    async def _record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self.stats.total_successes += 1
            self.stats.last_success_time = datetime.utcnow()

            if self.stats.state == CircuitState.HALF_OPEN:
                self.stats.success_count += 1

                # If we have enough successes, close the circuit
                if self.stats.success_count >= self.config.success_threshold:
                    logger.info(
                        "Circuit breaker closing",
                        circuit_name=self.name,
                        success_count=self.stats.success_count,
                    )
                    self.stats.state = CircuitState.CLOSED
                    self.stats.failure_count = 0
                    self.stats.success_count = 0
                    self.stats.opened_at = None
            elif self.stats.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.stats.failure_count = 0

    async def _record_failure(self) -> None:
        """Record a failed call."""
        async with self._lock:
            self.stats.total_failures += 1
            self.stats.last_failure_time = datetime.utcnow()
            self.stats.failure_count += 1

            if self.stats.state == CircuitState.HALF_OPEN:
                # Any failure in half-open opens the circuit
                logger.warning(
                    "Circuit breaker opening (failure in half-open)",
                    circuit_name=self.name,
                )
                self.stats.state = CircuitState.OPEN
                self.stats.opened_at = datetime.utcnow()
                self.stats.success_count = 0
            elif self.stats.state == CircuitState.CLOSED:
                # Check if we've exceeded failure threshold
                if self.stats.failure_count >= self.config.failure_threshold:
                    logger.warning(
                        "Circuit breaker opening (failure threshold exceeded)",
                        circuit_name=self.name,
                        failure_count=self.stats.failure_count,
                        threshold=self.config.failure_threshold,
                    )
                    self.stats.state = CircuitState.OPEN
                    self.stats.opened_at = datetime.utcnow()

    def get_stats(self) -> dict[str, Any]:
        """Get current circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.stats.state.value,
            "failure_count": self.stats.failure_count,
            "success_count": self.stats.success_count,
            "total_requests": self.stats.total_requests,
            "total_failures": self.stats.total_failures,
            "total_successes": self.stats.total_successes,
            "last_failure_time": self.stats.last_failure_time.isoformat() if self.stats.last_failure_time else None,
            "last_success_time": self.stats.last_success_time.isoformat() if self.stats.last_success_time else None,
            "opened_at": self.stats.opened_at.isoformat() if self.stats.opened_at else None,
        }

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        async def _reset():
            async with self._lock:
                logger.info("Circuit breaker manually reset", circuit_name=self.name)
                self.stats = CircuitBreakerStats()

        # Run reset in event loop if available
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_reset())
            else:
                loop.run_until_complete(_reset())
        except RuntimeError:
            # No event loop, create new one
            asyncio.run(_reset())


class CircuitBreakerManager:
    """Manages multiple circuit breakers."""

    def __init__(self):
        """Initialize circuit breaker manager."""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    def get_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker.

        Args:
            name: Circuit breaker name
            config: Optional configuration

        Returns:
            Circuit breaker instance
        """
        if name not in self._breakers:
            async def _create():
                async with self._lock:
                    if name not in self._breakers:
                        self._breakers[name] = CircuitBreaker(name, config)

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(_create())
                else:
                    loop.run_until_complete(_create())
            except RuntimeError:
                asyncio.run(_create())

        return self._breakers[name]

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {
            name: breaker.get_stats()
            for name, breaker in self._breakers.items()
        }


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()

