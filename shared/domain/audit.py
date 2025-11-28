"""
Audit Log System with Tamper-Evident Hash Chain

Implements immutable audit trail with cryptographic hash chaining
for compliance and security forensics.
"""

import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class AuditAction(str, Enum):
    """Types of auditable actions."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    AUTH_FAILURE = "auth_failure"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    EXPORT_DATA = "export_data"
    IMPORT_DATA = "import_data"
    CONFIG_CHANGE = "config_change"
    SECURITY_EVENT = "security_event"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditLogEntry(BaseModel):
    """
    Immutable audit log entry with tamper-evident hash chaining.

    Each entry contains a hash of its own content plus the previous entry's hash,
    creating a blockchain-like structure that detects tampering.
    """

    model_config = ConfigDict(frozen=True, validate_assignment=True)

    # Identity
    id: UUID = Field(default_factory=uuid4, description="Unique audit entry ID")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Event timestamp (UTC)"
    )

    # Action Details
    action: AuditAction = Field(..., description="Type of action performed")
    severity: AuditSeverity = Field(default=AuditSeverity.INFO)
    actor_id: UUID | None = Field(default=None, description="User/system performing action")
    actor_type: str = Field(default="user", description="Actor type (user/system/service)")

    # Resource Details
    resource_type: str = Field(..., description="Type of resource affected")
    resource_id: UUID | None = Field(default=None, description="Specific resource ID")
    resource_name: str | None = Field(default=None, max_length=200)

    # Event Details
    description: str = Field(..., max_length=1000, description="Human-readable description")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional structured data"
    )

    # Context
    ip_address: str | None = Field(default=None, max_length=45)
    user_agent: str | None = Field(default=None, max_length=500)
    session_id: UUID | None = Field(default=None)
    correlation_id: str | None = Field(default=None, description="Request correlation ID")

    # Changes (for UPDATE actions)
    old_values: dict[str, Any] | None = Field(default=None)
    new_values: dict[str, Any] | None = Field(default=None)

    # Tamper-Evident Chain
    previous_hash: str | None = Field(
        default=None, description="Hash of previous audit entry"
    )
    entry_hash: str = Field(..., description="Hash of this entry's content")

    @classmethod
    def create(
        cls,
        action: AuditAction,
        resource_type: str,
        description: str,
        previous_hash: str | None = None,
        **kwargs: Any,
    ) -> "AuditLogEntry":
        """
        Factory method to create a new audit log entry with automatic hashing.

        Args:
            action: Audit action type
            resource_type: Type of resource
            description: Event description
            previous_hash: Hash of previous entry in chain
            **kwargs: Additional fields

        Returns:
            AuditLogEntry: Immutable audit entry with computed hash
        """
        # Create entry without hash first
        entry_data = {
            "action": action,
            "resource_type": resource_type,
            "description": description,
            "previous_hash": previous_hash,
            **kwargs,
        }

        # Compute hash of entry content
        entry_hash = cls._compute_hash(entry_data)
        entry_data["entry_hash"] = entry_hash

        return cls(**entry_data)

    @staticmethod
    def _compute_hash(data: dict[str, Any]) -> str:
        """
        Compute SHA-256 hash of entry data.

        Args:
            data: Entry data dictionary

        Returns:
            str: Hex-encoded SHA-256 hash
        """
        # Create deterministic JSON string for hashing
        # Exclude the hash itself and ensure consistent ordering
        hashable_data = {
            k: v
            for k, v in data.items()
            if k not in ("entry_hash",) and v is not None
        }

        # Convert UUIDs and datetime to strings for JSON serialization
        def json_serializer(obj: Any) -> Any:
            if isinstance(obj, UUID):
                return str(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            return obj

        json_str = json.dumps(
            hashable_data, sort_keys=True, default=json_serializer, separators=(",", ":")
        )

        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def verify_hash(self) -> bool:
        """
        Verify that the entry's hash is correct (not tampered).

        Returns:
            bool: True if hash is valid
        """
        expected_hash = self._compute_hash(self.model_dump())
        return self.entry_hash == expected_hash

    def verify_chain(self, previous_entry: Optional["AuditLogEntry"]) -> bool:
        """
        Verify the chain integrity with the previous entry.

        Args:
            previous_entry: Previous audit log entry in chain

        Returns:
            bool: True if chain is valid
        """
        # First entry in chain
        if previous_entry is None:
            return self.previous_hash is None

        # Verify this entry's previous_hash matches previous entry's hash
        return self.previous_hash == previous_entry.entry_hash


class AuditLogChain:
    """
    Manages a chain of audit log entries with integrity verification.

    Provides methods to append entries and verify chain integrity.
    """

    def __init__(self):
        """Initialize empty audit log chain."""
        self.entries: list[AuditLogEntry] = []

    def append(
        self,
        action: AuditAction,
        resource_type: str,
        description: str,
        **kwargs: Any,
    ) -> AuditLogEntry:
        """
        Append a new entry to the chain.

        Args:
            action: Audit action
            resource_type: Resource type
            description: Event description
            **kwargs: Additional entry fields

        Returns:
            AuditLogEntry: Created entry
        """
        previous_hash = self.entries[-1].entry_hash if self.entries else None

        entry = AuditLogEntry.create(
            action=action,
            resource_type=resource_type,
            description=description,
            previous_hash=previous_hash,
            **kwargs,
        )

        self.entries.append(entry)
        return entry

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """
        Verify the integrity of the entire chain.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        if not self.entries:
            return True, []

        errors: list[str] = []

        # Verify first entry has no previous hash
        if self.entries[0].previous_hash is not None:
            errors.append("First entry should not have previous_hash")

        # Verify each entry's hash and chain link
        for i, entry in enumerate(self.entries):
            # Verify entry's own hash
            if not entry.verify_hash():
                errors.append(f"Entry {i} (ID: {entry.id}) has invalid hash")

            # Verify chain link (except first entry)
            if i > 0:
                previous_entry = self.entries[i - 1]
                if not entry.verify_chain(previous_entry):
                    errors.append(
                        f"Entry {i} (ID: {entry.id}) has broken chain link to previous entry"
                    )

        return len(errors) == 0, errors

    def get_entries_by_actor(self, actor_id: UUID) -> list[AuditLogEntry]:
        """Get all entries for a specific actor."""
        return [e for e in self.entries if e.actor_id == actor_id]

    def get_entries_by_resource(
        self, resource_type: str, resource_id: UUID | None = None
    ) -> list[AuditLogEntry]:
        """Get all entries for a specific resource."""
        if resource_id:
            return [
                e
                for e in self.entries
                if e.resource_type == resource_type and e.resource_id == resource_id
            ]
        return [e for e in self.entries if e.resource_type == resource_type]

    def get_entries_by_action(self, action: AuditAction) -> list[AuditLogEntry]:
        """Get all entries for a specific action type."""
        return [e for e in self.entries if e.action == action]

    def get_entries_in_timerange(
        self, start: datetime, end: datetime
    ) -> list[AuditLogEntry]:
        """Get all entries within a time range."""
        return [e for e in self.entries if start <= e.timestamp <= end]


class ComplianceChecker:
    """
    Compliance checking and reporting for audit logs.

    Analyzes audit logs for policy violations and compliance issues.
    """

    def __init__(self, audit_chain: AuditLogChain):
        """
        Initialize compliance checker.

        Args:
            audit_chain: Audit log chain to analyze
        """
        self.audit_chain = audit_chain

    def check_unauthorized_access_attempts(self) -> list[AuditLogEntry]:
        """
        Find unauthorized access attempts.

        Returns:
            List of audit entries with access denied events
        """
        return self.audit_chain.get_entries_by_action(AuditAction.ACCESS_DENIED)

    def check_failed_authentication_attempts(
        self, threshold: int = 5, window_minutes: int = 15
    ) -> dict[UUID, list[AuditLogEntry]]:
        """
        Find users with excessive failed authentication attempts.

        Args:
            threshold: Number of failures to trigger alert
            window_minutes: Time window for counting failures

        Returns:
            Dictionary mapping user IDs to their failed attempts
        """
        auth_failures = self.audit_chain.get_entries_by_action(
            AuditAction.AUTH_FAILURE
        )

        # Group by actor
        by_actor: dict[UUID, list[AuditLogEntry]] = {}
        for entry in auth_failures:
            if entry.actor_id:
                if entry.actor_id not in by_actor:
                    by_actor[entry.actor_id] = []
                by_actor[entry.actor_id].append(entry)

        # Filter to those exceeding threshold in window
        suspicious: dict[UUID, list[AuditLogEntry]] = {}
        for actor_id, entries in by_actor.items():
            if len(entries) < threshold:
                continue

            # Check if threshold exceeded within window
            sorted_entries = sorted(entries, key=lambda e: e.timestamp)
            for i in range(len(sorted_entries) - threshold + 1):
                window_start = sorted_entries[i].timestamp
                window_end = sorted_entries[i + threshold - 1].timestamp
                if (window_end - window_start).total_seconds() <= window_minutes * 60:
                    suspicious[actor_id] = entries
                    break

        return suspicious

    def check_data_exports(self) -> list[AuditLogEntry]:
        """
        Find all data export events for GDPR compliance.

        Returns:
            List of data export audit entries
        """
        return self.audit_chain.get_entries_by_action(AuditAction.EXPORT_DATA)

    def check_sensitive_resource_access(
        self, resource_type: str, resource_id: UUID
    ) -> list[AuditLogEntry]:
        """
        Audit trail for sensitive resource access.

        Args:
            resource_type: Resource type to audit
            resource_id: Specific resource ID

        Returns:
            List of all access events for the resource
        """
        return self.audit_chain.get_entries_by_resource(resource_type, resource_id)

    def generate_compliance_report(self) -> dict[str, Any]:
        """
        Generate comprehensive compliance report.

        Returns:
            Dictionary with compliance metrics and findings
        """
        is_valid, errors = self.audit_chain.verify_integrity()

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "total_entries": len(self.audit_chain.entries),
            "chain_integrity_valid": is_valid,
            "chain_integrity_errors": errors,
            "unauthorized_access_attempts": len(
                self.check_unauthorized_access_attempts()
            ),
            "suspicious_auth_failures": len(
                self.check_failed_authentication_attempts()
            ),
            "data_exports": len(self.check_data_exports()),
            "entries_by_severity": {
                severity.value: len(
                    [e for e in self.audit_chain.entries if e.severity == severity]
                )
                for severity in AuditSeverity
            },
            "entries_by_action": {
                action.value: len(
                    [e for e in self.audit_chain.entries if e.action == action]
                )
                for action in AuditAction
            },
        }

