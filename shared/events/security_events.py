"""
Security Domain Events

Events related to authentication, authorization, and security incidents.
"""

from datetime import datetime
from typing import ClassVar
from uuid import UUID

from pydantic import Field

from shared.events.base import DomainEvent


class UserAuthenticatedEvent(DomainEvent):
    """Event emitted when a user successfully authenticates."""

    EVENT_TYPE: ClassVar[str] = "security.auth.user_authenticated"

    user_id: UUID = Field(...)
    auth_strategy: str = Field(...)
    ip_address: str = Field(...)
    user_agent: str | None = Field(default=None)
    authenticated_at: datetime = Field(default_factory=datetime.utcnow)


class AuthenticationFailedEvent(DomainEvent):
    """Event emitted when authentication fails."""

    EVENT_TYPE: ClassVar[str] = "security.auth.authentication_failed"

    user_id: UUID | None = Field(default=None)
    email: str = Field(...)
    reason: str = Field(...)
    ip_address: str = Field(...)
    failed_at: datetime = Field(default_factory=datetime.utcnow)


class AccessGrantedEvent(DomainEvent):
    """Event emitted when access is granted to a resource."""

    EVENT_TYPE: ClassVar[str] = "security.access.granted"

    user_id: UUID = Field(...)
    resource_type: str = Field(...)
    resource_id: UUID = Field(...)
    action: str = Field(...)
    granted_at: datetime = Field(default_factory=datetime.utcnow)


class AccessDeniedEvent(DomainEvent):
    """Event emitted when access is denied to a resource."""

    EVENT_TYPE: ClassVar[str] = "security.access.denied"

    user_id: UUID = Field(...)
    resource_type: str = Field(...)
    resource_id: UUID = Field(...)
    action: str = Field(...)
    reason: str = Field(...)
    denied_at: datetime = Field(default_factory=datetime.utcnow)


class RoleAssignedEvent(DomainEvent):
    """Event emitted when a role is assigned to a user."""

    EVENT_TYPE: ClassVar[str] = "security.role.assigned"

    user_id: UUID = Field(...)
    role_id: UUID = Field(...)
    role_name: str = Field(...)
    assigned_by: UUID = Field(...)


class RoleRevokedEvent(DomainEvent):
    """Event emitted when a role is revoked from a user."""

    EVENT_TYPE: ClassVar[str] = "security.role.revoked"

    user_id: UUID = Field(...)
    role_id: UUID = Field(...)
    role_name: str = Field(...)
    revoked_by: UUID = Field(...)


class SecurityIncidentEvent(DomainEvent):
    """Event emitted when a security incident is detected."""

    EVENT_TYPE: ClassVar[str] = "security.incident.detected"

    incident_type: str = Field(...)
    severity: str = Field(..., description="low, medium, high, critical")
    description: str = Field(...)
    affected_user_id: UUID | None = Field(default=None)
    affected_resource_id: UUID | None = Field(default=None)
    ip_address: str | None = Field(default=None)
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    auto_mitigated: bool = Field(default=False)


class DataAccessEvent(DomainEvent):
    """Event emitted when sensitive data is accessed (GDPR compliance)."""

    EVENT_TYPE: ClassVar[str] = "security.data.accessed"

    user_id: UUID = Field(...)
    data_subject_id: UUID = Field(..., description="ID of person whose data was accessed")
    data_category: str = Field(..., description="Category of data (e.g., 'personal', 'grades')")
    purpose: str = Field(..., description="Purpose of access")
    legal_basis: str = Field(
        ..., description="Legal basis for access (consent, contract, legal_obligation, etc.)"
    )


class DataErasureRequestEvent(DomainEvent):
    """Event emitted when data erasure is requested (GDPR Right to be Forgotten)."""

    EVENT_TYPE: ClassVar[str] = "security.data.erasure_requested"

    data_subject_id: UUID = Field(..., description="Person requesting data erasure")
    requested_by: UUID = Field(..., description="User submitting request")
    request_date: datetime = Field(default_factory=datetime.utcnow)
    reason: str = Field(...)
    scope: str = Field(
        default="full", description="Scope of erasure (full, partial, pseudonymize)"
    )


class DataErasureCompletedEvent(DomainEvent):
    """Event emitted when data erasure is completed."""

    EVENT_TYPE: ClassVar[str] = "security.data.erasure_completed"

    data_subject_id: UUID = Field(...)
    completed_by: UUID = Field(...)
    completed_at: datetime = Field(default_factory=datetime.utcnow)
    records_affected: int = Field(...)
    pseudonymized: bool = Field(default=False)

