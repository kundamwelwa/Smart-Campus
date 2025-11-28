"""
Tamper-Evident Audit Logging

Implements immutable, hash-chained audit logs for compliance and security.
Each entry contains a hash of the previous entry, creating a tamper-evident chain.
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)


class AuditLogEntry(BaseModel):
    """
    Immutable audit log entry with hash chain for tamper-evidence.

    Each entry contains:
    - Event data
    - Hash of previous entry
    - Self hash

    This creates a blockchain-like structure that makes tampering detectable.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    id: UUID = Field(default_factory=uuid4)
    sequence_number: int = Field(..., ge=0, description="Sequential position in audit log")

    # Event Information
    event_type: str = Field(..., description="Type of audited event")
    action: str = Field(..., description="Action performed")
    resource_type: str = Field(..., description="Type of affected resource")
    resource_id: UUID | None = Field(default=None, description="Specific resource ID")

    # Actor Information
    user_id: UUID | None = Field(default=None, description="User who performed action")
    user_email: str | None = Field(default=None)
    service: str = Field(..., description="Service that generated the audit entry")

    # Context
    ip_address: str | None = Field(default=None)
    user_agent: str | None = Field(default=None)
    session_id: str | None = Field(default=None)

    # Status
    status: str = Field(..., description="success or failure")
    error_message: str | None = Field(default=None)

    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Additional Data
    details: dict[str, Any] = Field(default_factory=dict, description="Additional event details")

    # Tamper-Evidence (Hash Chain)
    previous_hash: str | None = Field(
        default=None, description="Hash of previous audit entry"
    )
    entry_hash: str = Field(..., description="Hash of this entry")

    @classmethod
    def create(
        cls,
        sequence_number: int,
        event_type: str,
        action: str,
        resource_type: str,
        service: str,
        status: str,
        resource_id: UUID | None = None,
        user_id: UUID | None = None,
        user_email: str | None = None,
        previous_hash: str | None = None,
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        **kwargs: Any,
    ) -> "AuditLogEntry":
        """
        Factory method to create audit log entry with computed hash.

        Args:
            sequence_number: Position in audit log
            event_type: Event type
            action: Action performed
            resource_type: Resource type
            service: Originating service
            status: Operation status
            previous_hash: Hash of previous entry (None for first entry)
            **kwargs: Additional fields

        Returns:
            AuditLogEntry: New audit log entry with hash
        """
        # Create entry without hash first
        entry_data = {
            "sequence_number": sequence_number,
            "event_type": event_type,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "user_id": user_id,
            "user_email": user_email,
            "service": service,
            "status": status,
            "previous_hash": previous_hash,
            "details": details or {},
            "ip_address": ip_address,
            "user_agent": user_agent,
            **kwargs,
        }

        # Compute hash
        entry_hash = cls._compute_hash(entry_data)

        # Create final entry
        return cls(**entry_data, entry_hash=entry_hash)

    @staticmethod
    def _compute_hash(entry_data: dict[str, Any]) -> str:
        """
        Compute SHA-256 hash of entry data.

        Args:
            entry_data: Entry data dictionary

        Returns:
            str: Hexadecimal hash string
        """
        # Create deterministic JSON representation
        # Exclude id and timestamp as they're generated
        hashable_data = {
            k: str(v) if isinstance(v, UUID) else v
            for k, v in entry_data.items()
            if k not in ("id", "entry_hash")
        }

        json_str = json.dumps(hashable_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def verify_hash(self) -> bool:
        """
        Verify that entry hash is valid.

        Returns:
            bool: True if hash is valid
        """
        data = self.model_dump(exclude={"id", "entry_hash"})
        computed_hash = self._compute_hash(data)
        return computed_hash == self.entry_hash

    def verify_chain(self, previous_entry: Optional["AuditLogEntry"]) -> bool:
        """
        Verify hash chain link with previous entry.

        Args:
            previous_entry: Previous audit log entry (None if this is first)

        Returns:
            bool: True if chain is valid
        """
        # First entry should have no previous hash
        if previous_entry is None:
            return self.previous_hash is None

        # Verify previous entry's hash matches our previous_hash field
        return self.previous_hash == previous_entry.entry_hash


class AuditLogger:
    """
    Audit logger service for creating and storing tamper-evident logs.

    Maintains the hash chain and ensures append-only semantics.
    """

    def __init__(self, storage_backend: Any):
        """
        Initialize audit logger.

        Args:
            storage_backend: Storage backend (MongoDB collection)
        """
        self.storage = storage_backend
        self._sequence_number = 0
        self._last_hash: str | None = None

    async def initialize(self) -> None:
        """Initialize audit logger and load last hash from storage."""
        # Get the latest audit entry to continue the chain
        latest = await self.storage.find_one(sort=[("sequence_number", -1)])

        if latest:
            self._sequence_number = latest["sequence_number"]
            self._last_hash = latest["entry_hash"]
            logger.info(
                "Audit logger initialized",
                last_sequence=self._sequence_number,
                last_hash=self._last_hash[:16],
            )
        else:
            logger.info("Audit logger initialized with empty chain")

    async def log(
        self,
        event_type: str,
        action: str,
        resource_type: str,
        service: str,
        status: str = "success",
        resource_id: UUID | None = None,
        user_id: UUID | None = None,
        user_email: str | None = None,
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> AuditLogEntry:
        """
        Create and store an audit log entry.

        Args:
            event_type: Event type
            action: Action performed
            resource_type: Resource type
            service: Service name
            status: Operation status
            **kwargs: Additional fields

        Returns:
            AuditLogEntry: Created audit entry
        """
        # Increment sequence
        self._sequence_number += 1

        # Create entry
        entry = AuditLogEntry.create(
            sequence_number=self._sequence_number,
            event_type=event_type,
            action=action,
            resource_type=resource_type,
            service=service,
            status=status,
            resource_id=resource_id,
            user_id=user_id,
            user_email=user_email,
            previous_hash=self._last_hash,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        # Store entry
        await self._store_entry(entry)

        # Update last hash
        self._last_hash = entry.entry_hash

        logger.info(
            "Audit entry created",
            sequence=entry.sequence_number,
            event_type=event_type,
            action=action,
            user_id=str(user_id) if user_id else None,
        )

        return entry

    async def _store_entry(self, entry: AuditLogEntry) -> None:
        """Store audit entry in database."""
        entry_dict = entry.model_dump(mode="json")
        entry_dict["_id"] = str(entry.id)

        await self.storage.insert_one(entry_dict)

    async def verify_chain_integrity(
        self, from_sequence: int = 0, to_sequence: int | None = None
    ) -> tuple[bool, list[str]]:
        """
        Verify integrity of audit log chain.

        Args:
            from_sequence: Starting sequence number
            to_sequence: Ending sequence number (None = all)

        Returns:
            Tuple of (is_valid, list of violation descriptions)
        """
        violations: list[str] = []

        query: dict[str, Any] = {"sequence_number": {"$gte": from_sequence}}
        if to_sequence is not None:
            query["sequence_number"]["$lte"] = to_sequence

        cursor = self.storage.find(query).sort("sequence_number", 1)

        previous_entry: AuditLogEntry | None = None

        async for doc in cursor:
            doc.pop("_id", None)
            entry = AuditLogEntry(**doc)

            # Verify entry hash
            if not entry.verify_hash():
                violations.append(
                    f"Sequence {entry.sequence_number}: Invalid entry hash"
                )

            # Verify chain link
            if not entry.verify_chain(previous_entry):
                violations.append(
                    f"Sequence {entry.sequence_number}: Broken hash chain"
                )

            previous_entry = entry

        is_valid = len(violations) == 0

        if is_valid:
            logger.info("Audit chain verification passed", from_sequence=from_sequence)
        else:
            logger.warning(
                "Audit chain verification failed",
                violations_count=len(violations),
                violations=violations,
            )

        return is_valid, violations

    async def get_entries(
        self,
        from_timestamp: datetime | None = None,
        to_timestamp: datetime | None = None,
        user_id: UUID | None = None,
        resource_type: str | None = None,
        action: str | None = None,
        limit: int = 100,
    ) -> list[AuditLogEntry]:
        """
        Query audit log entries with filters.

        Args:
            from_timestamp: Start time
            to_timestamp: End time
            user_id: Filter by user
            resource_type: Filter by resource type
            action: Filter by action
            limit: Maximum results

        Returns:
            List of audit log entries
        """
        query: dict[str, Any] = {}

        if from_timestamp or to_timestamp:
            query["timestamp"] = {}
            if from_timestamp:
                query["timestamp"]["$gte"] = from_timestamp
            if to_timestamp:
                query["timestamp"]["$lte"] = to_timestamp

        if user_id:
            query["user_id"] = str(user_id)

        if resource_type:
            query["resource_type"] = resource_type

        if action:
            query["action"] = action

        cursor = self.storage.find(query).sort("sequence_number", -1).limit(limit)

        entries = []
        async for doc in cursor:
            doc.pop("_id", None)
            entries.append(AuditLogEntry(**doc))

        return entries

