"""
Core Entity Hierarchy

Implements the foundational 5+ level inheritance hierarchy:
AbstractEntity → VersionedEntity → AuditableEntity → Person → Student/Lecturer/Staff/Guest/Admin

Features:
- Universal ID and lifecycle management
- Schema versioning and migration support
- Audit trail integration
- Dynamic role attachments at runtime
- Immutability patterns where appropriate
"""

from abc import ABC, abstractmethod
from datetime import date, datetime
from enum import Enum
from typing import Any, ClassVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EntityStatus(str, Enum):
    """Entity lifecycle status."""

    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    DELETED = "deleted"


class AbstractEntity(BaseModel, ABC):
    """
    Base abstract entity class providing universal ID and lifecycle management.

    All domain entities inherit from this class, establishing a consistent
    identity and lifecycle pattern across the entire system.

    Level 1 of inheritance hierarchy.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        arbitrary_types_allowed=True,
    )

    id: UUID = Field(default_factory=uuid4, description="Universal unique identifier")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Entity creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )
    status: EntityStatus = Field(
        default=EntityStatus.ACTIVE, description="Current lifecycle status"
    )

    def __hash__(self) -> int:
        """Hash based on entity ID for set/dict usage."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on entity ID and type."""
        if not isinstance(other, AbstractEntity):
            return NotImplemented
        return self.id == other.id and isinstance(other, type(self))

    @abstractmethod
    def validate_business_rules(self) -> bool:
        """
        Validate entity-specific business rules.

        Returns:
            bool: True if all business rules are satisfied

        Raises:
            ValueError: If business rules are violated
        """

    def mark_updated(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()

    def activate(self) -> None:
        """Activate the entity."""
        self.status = EntityStatus.ACTIVE
        self.mark_updated()

    def deactivate(self) -> None:
        """Deactivate the entity."""
        self.status = EntityStatus.INACTIVE
        self.mark_updated()

    def archive(self) -> None:
        """Archive the entity."""
        self.status = EntityStatus.ARCHIVED
        self.mark_updated()

    def soft_delete(self) -> None:
        """Soft delete the entity (GDPR-compliant)."""
        self.status = EntityStatus.DELETED
        self.mark_updated()


class VersionedEntity(AbstractEntity, ABC):
    """
    Entity with schema versioning support.

    Enables schema evolution and backward compatibility through version tracking.
    Supports migration between versions.

    Level 2 of inheritance hierarchy.
    """

    schema_version: int = Field(default=1, description="Entity schema version")
    migration_history: list[str] = Field(
        default_factory=list, description="Applied migration identifiers"
    )

    CURRENT_SCHEMA_VERSION: ClassVar[int] = 1

    @classmethod
    def get_schema_version(cls) -> int:
        """Get the current schema version for this entity type."""
        return cls.CURRENT_SCHEMA_VERSION

    def needs_migration(self) -> bool:
        """Check if entity needs schema migration."""
        return self.schema_version < self.CURRENT_SCHEMA_VERSION

    def migrate_to_latest(self) -> None:
        """
        Migrate entity to latest schema version.

        Subclasses should override this to implement specific migrations.
        """
        if not self.needs_migration():
            return

        # Record migration
        migration_id = f"v{self.schema_version}_to_v{self.CURRENT_SCHEMA_VERSION}"
        self.migration_history.append(migration_id)
        self.schema_version = self.CURRENT_SCHEMA_VERSION
        self.mark_updated()

    def get_migration_history(self) -> list[str]:
        """Get the full migration history."""
        return self.migration_history.copy()


class AuditableEntity(VersionedEntity, ABC):
    """
    Entity with comprehensive audit trail support.

    Tracks creation, modification, and access patterns for compliance
    and security purposes. Integrates with tamper-evident audit log system.

    Level 3 of inheritance hierarchy.
    """

    created_by: UUID | None = Field(default=None, description="Creator user ID")
    updated_by: UUID | None = Field(default=None, description="Last updater user ID")
    access_count: int = Field(default=0, description="Number of times accessed")
    last_accessed_at: datetime | None = Field(
        default=None, description="Last access timestamp"
    )
    last_accessed_by: UUID | None = Field(default=None, description="Last accessor user ID")

    def record_creation(self, user_id: UUID) -> None:
        """Record entity creation by a user."""
        self.created_by = user_id
        self.updated_by = user_id

    def record_update(self, user_id: UUID) -> None:
        """Record entity update by a user."""
        self.updated_by = user_id
        self.mark_updated()

    def record_access(self, user_id: UUID) -> None:
        """Record entity access by a user."""
        self.access_count += 1
        self.last_accessed_at = datetime.utcnow()
        self.last_accessed_by = user_id

    def get_audit_summary(self) -> dict[str, Any]:
        """Get audit summary for this entity."""
        return {
            "entity_id": str(self.id),
            "entity_type": self.__class__.__name__,
            "created_at": self.created_at.isoformat(),
            "created_by": str(self.created_by) if self.created_by else None,
            "updated_at": self.updated_at.isoformat(),
            "updated_by": str(self.updated_by) if self.updated_by else None,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at.isoformat()
            if self.last_accessed_at
            else None,
            "schema_version": self.schema_version,
        }


class Person(AuditableEntity, ABC):
    """
    Abstract base class for all person entities in the system.

    Supports dynamic role attachments at runtime, allowing flexible
    role-based access control and multi-role users (e.g., Student who is also a TA).

    Level 4 of inheritance hierarchy.
    """

    # Personal Information
    email: str = Field(..., description="Primary email address")
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    middle_name: str | None = Field(default=None, max_length=100)
    date_of_birth: date | None = Field(default=None)
    phone_number: str | None = Field(default=None, max_length=20)

    # Dynamic Role Attachments
    attached_roles: list[str] = Field(
        default_factory=list,
        description="Dynamically attached roles (e.g., 'TA', 'Researcher')",
    )

    # Privacy & GDPR
    is_pseudonymized: bool = Field(
        default=False, description="Whether personal data has been pseudonymized"
    )
    consent_given: bool = Field(default=False, description="User consent for data processing")
    data_retention_until: datetime | None = Field(
        default=None, description="Data retention expiry date"
    )

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email validation."""
        if "@" not in v or "." not in v:
            raise ValueError("Invalid email format")
        return v.lower()

    @property
    def full_name(self) -> str:
        """Get formatted full name."""
        if self.middle_name:
            return f"{self.first_name} {self.middle_name} {self.last_name}"
        return f"{self.first_name} {self.last_name}"

    @property
    def display_name(self) -> str:
        """Get display name (pseudonymized if applicable)."""
        if self.is_pseudonymized:
            return f"User-{self.id.hex[:8]}"
        return self.full_name

    def attach_role(self, role: str) -> None:
        """
        Dynamically attach a new role to this person at runtime.

        Args:
            role: Role identifier to attach (e.g., 'TA', 'Researcher')
        """
        if role not in self.attached_roles:
            self.attached_roles.append(role)
            self.mark_updated()

    def detach_role(self, role: str) -> None:
        """Remove a dynamically attached role."""
        if role in self.attached_roles:
            self.attached_roles.remove(role)
            self.mark_updated()

    def has_role(self, role: str) -> bool:
        """Check if person has a specific attached role."""
        return role in self.attached_roles

    def pseudonymize(self) -> None:
        """
        Pseudonymize personal data for GDPR compliance.

        Replaces identifiable information with anonymized placeholders
        while maintaining referential integrity for analytics.
        """
        self.first_name = "User"
        self.last_name = self.id.hex[:8]
        self.middle_name = None
        self.email = f"user-{self.id.hex[:8]}@pseudonymized.local"
        self.phone_number = None
        self.date_of_birth = None
        self.is_pseudonymized = True
        self.mark_updated()


class Student(Person):
    """
    Student entity representing enrolled learners.

    Level 5 of inheritance hierarchy (5+ levels achieved).
    """

    student_id: str = Field(..., description="University student ID")
    enrollment_date: date = Field(default_factory=date.today)
    expected_graduation_date: date | None = Field(default=None)
    major: str | None = Field(default=None, max_length=100)
    minor: str | None = Field(default=None, max_length=100)
    gpa: float = Field(default=0.0, ge=0.0, le=4.0)
    credits_earned: int = Field(default=0, ge=0)
    academic_standing: str = Field(
        default="good", description="Academic standing status (good/probation/suspended)"
    )

    def validate_business_rules(self) -> bool:
        """Validate student-specific business rules."""
        if self.gpa < 0 or self.gpa > 4.0:
            raise ValueError("GPA must be between 0.0 and 4.0")
        if self.credits_earned < 0:
            raise ValueError("Credits earned cannot be negative")
        return True

    def update_gpa(self, new_gpa: float) -> None:
        """Update student GPA with validation."""
        if new_gpa < 0.0 or new_gpa > 4.0:
            raise ValueError("GPA must be between 0.0 and 4.0")
        self.gpa = new_gpa
        self._update_academic_standing()
        self.mark_updated()

    def add_credits(self, credits: int) -> None:
        """Add earned credits."""
        if credits < 0:
            raise ValueError("Cannot add negative credits")
        self.credits_earned += credits
        self.mark_updated()

    def _update_academic_standing(self) -> None:
        """Update academic standing based on GPA."""
        if self.gpa >= 2.0:
            self.academic_standing = "good"
        elif self.gpa >= 1.5:
            self.academic_standing = "probation"
        else:
            self.academic_standing = "suspended"


class Lecturer(Person):
    """
    Lecturer/Professor entity representing teaching staff.

    Level 5 of inheritance hierarchy.
    """

    employee_id: str = Field(..., description="Employee identifier")
    department: str = Field(..., max_length=100)
    title: str = Field(default="Lecturer", max_length=100)
    office_location: str | None = Field(default=None, max_length=200)
    specialization: list[str] = Field(default_factory=list)
    tenure_status: bool = Field(default=False)
    hire_date: date = Field(default_factory=date.today)
    max_course_load: int = Field(default=4, ge=1, le=8, description="Max courses per semester")

    def validate_business_rules(self) -> bool:
        """Validate lecturer-specific business rules."""
        if self.max_course_load < 1 or self.max_course_load > 8:
            raise ValueError("Course load must be between 1 and 8")
        return True


class Staff(Person):
    """
    Administrative and support staff entity.

    Level 5 of inheritance hierarchy.
    """

    employee_id: str = Field(..., description="Employee identifier")
    department: str = Field(..., max_length=100)
    job_title: str = Field(..., max_length=100)
    office_location: str | None = Field(default=None, max_length=200)
    hire_date: date = Field(default_factory=date.today)
    clearance_level: int = Field(default=1, ge=1, le=5, description="Security clearance level")

    def validate_business_rules(self) -> bool:
        """Validate staff-specific business rules."""
        if self.clearance_level < 1 or self.clearance_level > 5:
            raise ValueError("Clearance level must be between 1 and 5")
        return True


class Guest(Person):
    """
    Guest/Visitor entity with temporary access.

    Level 5 of inheritance hierarchy.
    """

    visitor_id: str = Field(..., description="Temporary visitor ID")
    host_id: UUID | None = Field(default=None, description="Host person ID")
    visit_purpose: str = Field(..., max_length=500)
    valid_from: datetime = Field(default_factory=datetime.utcnow)
    valid_until: datetime = Field(..., description="Access expiry time")
    sponsored_by: UUID | None = Field(default=None, description="Sponsor user ID")

    def validate_business_rules(self) -> bool:
        """Validate guest-specific business rules."""
        if self.valid_until <= self.valid_from:
            raise ValueError("Valid until must be after valid from")
        return True

    def is_access_valid(self) -> bool:
        """Check if guest access is still valid."""
        now = datetime.utcnow()
        return self.valid_from <= now <= self.valid_until and self.status == EntityStatus.ACTIVE

    def extend_access(self, new_valid_until: datetime) -> None:
        """Extend guest access period."""
        if new_valid_until <= self.valid_until:
            raise ValueError("New expiry must be later than current expiry")
        self.valid_until = new_valid_until
        self.mark_updated()


class Admin(Person):
    """
    Administrator entity with elevated privileges.

    Level 5 of inheritance hierarchy.
    """

    employee_id: str = Field(..., description="Employee identifier")
    admin_level: int = Field(default=1, ge=1, le=3, description="Admin privilege level")
    department: str = Field(default="IT", max_length=100)
    can_manage_users: bool = Field(default=True)
    can_manage_courses: bool = Field(default=False)
    can_manage_facilities: bool = Field(default=False)
    can_manage_security: bool = Field(default=False)
    can_access_audit_logs: bool = Field(default=True)
    two_factor_enabled: bool = Field(default=True)

    def validate_business_rules(self) -> bool:
        """Validate admin-specific business rules."""
        if self.admin_level < 1 or self.admin_level > 3:
            raise ValueError("Admin level must be between 1 and 3")
        if not self.two_factor_enabled:
            raise ValueError("Two-factor authentication is mandatory for admins")
        return True

    def has_permission(self, resource: str) -> bool:
        """
        Check if admin has permission to manage a specific resource.

        Args:
            resource: Resource type (users/courses/facilities/security)

        Returns:
            bool: True if admin has permission
        """
        permission_map = {
            "users": self.can_manage_users,
            "courses": self.can_manage_courses,
            "facilities": self.can_manage_facilities,
            "security": self.can_manage_security,
            "audit_logs": self.can_access_audit_logs,
        }
        return permission_map.get(resource, False) or self.admin_level == 3  # Level 3 = super admin


# Demonstration of deep inheritance (6 levels):
# AbstractEntity (1) → VersionedEntity (2) → AuditableEntity (3) → Person (4) → Student (5)
# Plus potential for Student → GraduateStudent (6) for even deeper hierarchy

