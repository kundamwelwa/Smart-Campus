"""
Concurrency Control: Optimistic and Pessimistic Locking

Provides both optimistic concurrency control (version-based) and
pessimistic locking (explicit locks) for different use cases.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import structlog

from shared.events.store import ConcurrencyError

logger = structlog.get_logger(__name__)


class Lock:
    """
    Pessimistic lock for resource protection.

    Provides exclusive access to a resource with timeout support.
    """

    def __init__(
        self,
        resource_id: str,
        lock_id: UUID,
        owner: str,
        expires_at: datetime,
    ):
        """
        Initialize lock.

        Args:
            resource_id: Resource being locked
            lock_id: Unique lock identifier
            owner: Lock owner identifier
            expires_at: Lock expiration time
        """
        self.resource_id = resource_id
        self.lock_id = lock_id
        self.owner = owner
        self.expires_at = expires_at
        self.acquired_at = datetime.utcnow()

    def is_expired(self) -> bool:
        """Check if lock has expired."""
        return datetime.utcnow() > self.expires_at

    def time_remaining(self) -> timedelta:
        """Get time remaining until expiration."""
        return self.expires_at - datetime.utcnow()


class LockManager:
    """
    Manages pessimistic locks for resources.

    Thread-safe lock management with expiration and cleanup.
    """

    def __init__(self):
        """Initialize lock manager."""
        self._locks: dict[str, Lock] = {}
        self._lock = asyncio.Lock()  # Protects _locks dictionary
        logger.info("Lock manager initialized")

    async def acquire_lock(
        self,
        resource_id: str,
        owner: str,
        timeout_seconds: int = 30,
        wait_timeout: float | None = None,
    ) -> Lock | None:
        """
        Acquire a pessimistic lock on a resource.

        Args:
            resource_id: Resource to lock
            owner: Lock owner identifier
            timeout_seconds: Lock duration in seconds
            wait_timeout: Maximum time to wait if lock is held (None = fail immediately)

        Returns:
            Lock instance if acquired, None if failed
        """
        async with self._lock:
            # Clean up expired locks
            await self._cleanup_expired_locks()

            # Check if resource is already locked
            existing_lock = self._locks.get(resource_id)

            if existing_lock and not existing_lock.is_expired():
                # Lock is held - wait or fail
                if wait_timeout is None:
                    logger.warning(
                        "Lock acquisition failed - resource already locked",
                        resource_id=resource_id,
                        owner=owner,
                        current_owner=existing_lock.owner,
                    )
                    return None

                # Wait for lock to be released
                start_time = datetime.utcnow()
                while existing_lock and not existing_lock.is_expired():
                    elapsed = (datetime.utcnow() - start_time).total_seconds()
                    if elapsed >= wait_timeout:
                        logger.warning(
                            "Lock acquisition timeout",
                            resource_id=resource_id,
                            owner=owner,
                            wait_timeout=wait_timeout,
                        )
                        return None

                    # Release lock and wait
                    await asyncio.sleep(0.1)
                    await self._cleanup_expired_locks()
                    existing_lock = self._locks.get(resource_id)

            # Acquire lock
            from uuid import uuid4

            lock = Lock(
                resource_id=resource_id,
                lock_id=uuid4(),
                owner=owner,
                expires_at=datetime.utcnow() + timedelta(seconds=timeout_seconds),
            )

            self._locks[resource_id] = lock

            logger.info(
                "Lock acquired",
                resource_id=resource_id,
                owner=owner,
                lock_id=str(lock.lock_id),
                expires_at=lock.expires_at.isoformat(),
            )

            return lock

    async def release_lock(self, resource_id: str, owner: str) -> bool:
        """
        Release a lock on a resource.

        Args:
            resource_id: Resource to unlock
            owner: Lock owner (must match)

        Returns:
            True if lock was released, False if not found or owner mismatch
        """
        async with self._lock:
            lock = self._locks.get(resource_id)

            if not lock:
                logger.warning("Lock not found", resource_id=resource_id)
                return False

            if lock.owner != owner:
                logger.warning(
                    "Lock owner mismatch",
                    resource_id=resource_id,
                    owner=owner,
                    lock_owner=lock.owner,
                )
                return False

            del self._locks[resource_id]

            logger.info(
                "Lock released",
                resource_id=resource_id,
                owner=owner,
                lock_id=str(lock.lock_id),
            )

            return True

    async def extend_lock(
        self, resource_id: str, owner: str, additional_seconds: int
    ) -> bool:
        """
        Extend lock expiration time.

        Args:
            resource_id: Resource ID
            owner: Lock owner
            additional_seconds: Additional seconds to add

        Returns:
            True if lock was extended, False otherwise
        """
        async with self._lock:
            lock = self._locks.get(resource_id)

            if not lock or lock.owner != owner:
                return False

            if lock.is_expired():
                del self._locks[resource_id]
                return False

            lock.expires_at += timedelta(seconds=additional_seconds)

            logger.info(
                "Lock extended",
                resource_id=resource_id,
                owner=owner,
                new_expires_at=lock.expires_at.isoformat(),
            )

            return True

    async def is_locked(self, resource_id: str) -> bool:
        """Check if resource is currently locked."""
        async with self._lock:
            await self._cleanup_expired_locks()
            lock = self._locks.get(resource_id)
            return lock is not None and not lock.is_expired()

    async def get_lock_info(self, resource_id: str) -> dict[str, Any] | None:
        """Get information about current lock."""
        async with self._lock:
            await self._cleanup_expired_locks()
            lock = self._locks.get(resource_id)

            if not lock or lock.is_expired():
                return None

            return {
                "resource_id": lock.resource_id,
                "lock_id": str(lock.lock_id),
                "owner": lock.owner,
                "acquired_at": lock.acquired_at.isoformat(),
                "expires_at": lock.expires_at.isoformat(),
                "time_remaining_seconds": lock.time_remaining().total_seconds(),
            }

    async def _cleanup_expired_locks(self) -> None:
        """Remove expired locks."""
        expired = [
            resource_id
            for resource_id, lock in self._locks.items()
            if lock.is_expired()
        ]

        for resource_id in expired:
            del self._locks[resource_id]
            logger.debug("Expired lock removed", resource_id=resource_id)

    async def get_all_locks(self) -> dict[str, dict[str, Any]]:
        """Get information about all active locks."""
        async with self._lock:
            await self._cleanup_expired_locks()

            return {
                resource_id: {
                    "lock_id": str(lock.lock_id),
                    "owner": lock.owner,
                    "acquired_at": lock.acquired_at.isoformat(),
                    "expires_at": lock.expires_at.isoformat(),
                }
                for resource_id, lock in self._locks.items()
            }


# Global lock manager instance
_lock_manager: LockManager | None = None


def get_lock_manager() -> LockManager:
    """
    Get or create global lock manager.

    Returns:
        LockManager instance
    """
    global _lock_manager

    if _lock_manager is None:
        _lock_manager = LockManager()

    return _lock_manager


class OptimisticConcurrencyControl:
    """
    Optimistic concurrency control using version numbers.

    Used for read-heavy workloads where conflicts are rare.
    """

    @staticmethod
    def check_version(
        expected_version: int, current_version: int, resource_id: str
    ) -> None:
        """
        Check if version matches expected value.

        Args:
            expected_version: Expected version number
            current_version: Current version number
            resource_id: Resource identifier (for error message)

        Raises:
            ConcurrencyError: If versions don't match
        """
        if expected_version != current_version:
            raise ConcurrencyError(
                f"Optimistic concurrency conflict for {resource_id}: "
                f"expected version {expected_version}, but current is {current_version}"
            )

    @staticmethod
    def increment_version(current_version: int) -> int:
        """Increment version number."""
        return current_version + 1

