"""
Security Domain Models

Entities for authentication, authorization, roles, permissions, and access control.
Supports pluggable auth strategies, RBAC, and ABAC.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from shared.domain.entities import VersionedEntity


class AuthStrategy(str, Enum):
    """Authentication strategy type."""

    PASSWORD = "password"
    OAUTH = "oauth"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    TWO_FACTOR = "two_factor"
    SSO = "sso"


class PermissionAction(str, Enum):
    """CRUD permission actions."""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    APPROVE = "approve"


class ResourceType(str, Enum):
    """Protected resource types."""

    USER = "user"
    STUDENT = "student"
    COURSE = "course"
    ENROLLMENT = "enrollment"
    GRADE = "grade"
    ROOM = "room"
    FACILITY = "facility"
    BOOKING = "booking"
    AUDIT_LOG = "audit_log"
    SECURITY_INCIDENT = "security_incident"


class Credential(BaseModel, ABC):
    """
    Abstract base credential for pluggable authentication strategies.

    Supports multiple auth methods through strategy pattern.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: UUID = Field(..., description="Associated user ID")
    strategy: AuthStrategy = Field(...)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used_at: datetime | None = Field(default=None)
    expires_at: datetime | None = Field(default=None)

    @abstractmethod
    def verify(self, challenge: Any) -> bool:
        """
        Verify the credential against a challenge.

        Args:
            challenge: Authentication challenge (password, token, etc.)

        Returns:
            bool: True if verification succeeds
        """

    def is_expired(self) -> bool:
        """Check if credential has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def mark_used(self) -> None:
        """Record credential usage."""
        self.last_used_at = datetime.utcnow()


class PasswordCredential(Credential):
    """Password-based authentication credential."""

    strategy: AuthStrategy = Field(default=AuthStrategy.PASSWORD)
    password_hash: str = Field(..., description="Hashed password (bcrypt)")
    salt: str = Field(..., description="Password salt")
    password_changed_at: datetime = Field(default_factory=datetime.utcnow)
    must_change_password: bool = Field(default=False)
    failed_attempts: int = Field(default=0, ge=0)
    locked_until: datetime | None = Field(default=None)

    def verify(self, challenge: Any) -> bool:
        """
        Verify password credential.

        Note: Actual password hashing/verification happens in auth service.
        This is a domain model placeholder.
        """
        # This will be implemented in the auth service with passlib
        return True  # Placeholder

    def is_locked(self) -> bool:
        """Check if account is locked due to failed attempts."""
        if self.locked_until is None:
            return False
        return datetime.utcnow() < self.locked_until

    def record_failed_attempt(self) -> None:
        """Record failed authentication attempt."""
        self.failed_attempts += 1
        if self.failed_attempts >= 5:
            # Lock account for 30 minutes
            self.locked_until = datetime.utcnow() + timedelta(minutes=30)

    def reset_failed_attempts(self) -> None:
        """Reset failed attempt counter after successful auth."""
        self.failed_attempts = 0
        self.locked_until = None


class OAuthCredential(Credential):
    """OAuth-based authentication credential."""

    strategy: AuthStrategy = Field(default=AuthStrategy.OAUTH)
    provider: str = Field(..., description="OAuth provider (google, microsoft, github, etc.)")
    provider_user_id: str = Field(..., description="User ID from OAuth provider")
    access_token: str | None = Field(default=None)
    refresh_token: str | None = Field(default=None)
    token_expires_at: datetime | None = Field(default=None)

    def verify(self, challenge: Any) -> bool:
        """Verify OAuth credential (token validation)."""
        # Implemented in auth service
        return True  # Placeholder


class CertificateCredential(Credential):
    """
    Certificate-based authentication credential.

    Supports X.509 certificates for strong authentication.
    """

    strategy: AuthStrategy = Field(default=AuthStrategy.CERTIFICATE)
    certificate_serial: str = Field(..., description="Certificate serial number")
    certificate_thumbprint: str = Field(..., description="SHA-256 thumbprint")
    issuer: str = Field(..., description="Certificate authority (CA)")
    subject: str = Field(..., description="Certificate subject (CN)")
    valid_from: datetime = Field(..., description="Certificate validity start")
    valid_until: datetime = Field(..., description="Certificate validity end")
    certificate_pem: str | None = Field(
        default=None, description="PEM-encoded certificate (encrypted storage)"
    )

    def verify(self, challenge: Any) -> bool:
        """
        Verify certificate credential.

        Args:
            challenge: Certificate or signature to verify

        Returns:
            bool: True if certificate is valid
        """
        # Certificate verification happens in auth service
        # This checks expiration and basic validity
        now = datetime.utcnow()
        return not (now < self.valid_from or now > self.valid_until)  # Placeholder - actual verification in auth service

    def is_certificate_valid(self) -> bool:
        """Check if certificate is within validity period."""
        now = datetime.utcnow()
        return self.valid_from <= now <= self.valid_until


class Permission(BaseModel):
    """
    Fine-grained permission for RBAC/ABAC.

    Represents a specific action on a specific resource type.
    """

    model_config = ConfigDict(frozen=True)

    action: PermissionAction = Field(...)
    resource_type: ResourceType = Field(...)
    resource_id: UUID | None = Field(
        default=None, description="Specific resource ID (None = all)"
    )
    conditions: dict[str, Any] = Field(
        default_factory=dict, description="ABAC attribute conditions"
    )

    def __str__(self) -> str:
        """String representation of permission."""
        if self.resource_id:
            return f"{self.action.value}:{self.resource_type.value}:{self.resource_id}"
        return f"{self.action.value}:{self.resource_type.value}"

    def matches(
        self, action: PermissionAction, resource_type: ResourceType, resource_id: UUID | None = None
    ) -> bool:
        """
        Check if this permission matches the requested action.

        Args:
            action: Requested action
            resource_type: Requested resource type
            resource_id: Specific resource ID (optional)

        Returns:
            bool: True if permission matches
        """
        if self.action != action:
            return False
        if self.resource_type != resource_type:
            return False
        # If permission is for all resources (resource_id is None), it matches
        if self.resource_id is None:
            return True
        # Otherwise, check specific resource ID
        return self.resource_id == resource_id


class Role(VersionedEntity):
    """
    Role entity for RBAC (Role-Based Access Control).

    Roles group permissions and can be hierarchical.
    """

    name: str = Field(..., min_length=1, max_length=100, description="Role name")
    description: str = Field(..., max_length=500)
    permissions: list[Permission] = Field(default_factory=list)
    inherits_from: list[UUID] = Field(
        default_factory=list, description="Parent role IDs for inheritance"
    )
    priority: int = Field(default=0, description="Role priority for conflict resolution")
    is_system_role: bool = Field(
        default=False, description="System-defined role (cannot be deleted)"
    )

    def validate_business_rules(self) -> bool:
        """Validate role business rules."""
        return True

    def has_permission(
        self,
        action: PermissionAction,
        resource_type: ResourceType,
        resource_id: UUID | None = None,
    ) -> bool:
        """
        Check if role has a specific permission.

        Args:
            action: Permission action
            resource_type: Resource type
            resource_id: Specific resource ID (optional)

        Returns:
            bool: True if role has permission
        """
        return any(
            perm.matches(action, resource_type, resource_id) for perm in self.permissions
        )

    def add_permission(self, permission: Permission) -> None:
        """Add a permission to this role."""
        if permission not in self.permissions:
            self.permissions.append(permission)
            self.mark_updated()

    def remove_permission(self, permission: Permission) -> None:
        """Remove a permission from this role."""
        if permission in self.permissions:
            self.permissions.remove(permission)
            self.mark_updated()


class AuthToken(BaseModel):
    """
    Authentication token (JWT-based).

    Represents an active session token for authenticated users.
    """

    model_config = ConfigDict(frozen=True)

    token_id: UUID = Field(default_factory=UUID, description="Token unique ID (jti)")
    user_id: UUID = Field(..., description="Token owner user ID")
    token_type: str = Field(default="access", description="Token type (access/refresh)")
    issued_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(...)
    scopes: list[str] = Field(default_factory=list, description="Token scopes")
    ip_address: str | None = Field(default=None, description="Issuing IP address")
    user_agent: str | None = Field(default=None, description="Issuing user agent")

    def is_expired(self) -> bool:
        """Check if token has expired."""
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        """Check if token is valid (not expired)."""
        return not self.is_expired()

    @classmethod
    def create_access_token(
        cls, user_id: UUID, expires_in_minutes: int = 30, scopes: list[str] | None = None
    ) -> "AuthToken":
        """
        Create an access token.

        Args:
            user_id: User UUID
            expires_in_minutes: Token validity period
            scopes: Token scopes

        Returns:
            AuthToken: New access token
        """
        return cls(
            user_id=user_id,
            token_type="access",
            expires_at=datetime.utcnow() + timedelta(minutes=expires_in_minutes),
            scopes=scopes or [],
        )

    @classmethod
    def create_refresh_token(
        cls, user_id: UUID, expires_in_days: int = 7
    ) -> "AuthToken":
        """
        Create a refresh token.

        Args:
            user_id: User UUID
            expires_in_days: Token validity period

        Returns:
            AuthToken: New refresh token
        """
        return cls(
            user_id=user_id,
            token_type="refresh",
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days),
            scopes=["refresh"],
        )

