"""
User Service Database Models

SQLAlchemy ORM models for user data persistence.
"""

import enum
from datetime import date, datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, Integer, String
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from shared.database.postgres import Base


class UserType(str, enum.Enum):
    """User type enumeration."""

    STUDENT = "student"
    LECTURER = "lecturer"
    STAFF = "staff"
    GUEST = "guest"
    ADMIN = "admin"


class UserModel(Base):
    """
    User database model.

    Stores core user information and authentication credentials.
    """

    __tablename__ = "users"

    # Primary Key
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)

    # Basic Info
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    middle_name: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Authentication
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    two_factor_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # User Type
    user_type: Mapped[UserType] = mapped_column(
        SQLEnum(UserType, name="user_type_enum"), nullable=False
    )

    # Roles (JSON array of role IDs)
    role_ids: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    attached_roles: Mapped[list] = mapped_column(
        JSON, default=list, nullable=False, comment="Dynamically attached roles"
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Privacy & GDPR
    is_pseudonymized: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    consent_given: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Status
    status: Mapped[str] = mapped_column(
        String(20), default="active", nullable=False, index=True
    )

    def __repr__(self) -> str:
        """String representation."""
        return f"<UserModel(id={self.id}, email={self.email}, type={self.user_type})>"


class StudentModel(Base):
    """
    Student-specific data model.

    Extends user data with student-specific fields.
    """

    __tablename__ = "students"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), unique=True, index=True, nullable=False
    )

    student_id: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    enrollment_date: Mapped[date] = mapped_column(Date, nullable=False)
    expected_graduation_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    major: Mapped[str | None] = mapped_column(String(100), nullable=True)
    minor: Mapped[str | None] = mapped_column(String(100), nullable=True)
    gpa: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    credits_earned: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    academic_standing: Mapped[str] = mapped_column(String(20), default="good", nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class LecturerModel(Base):
    """Lecturer-specific data model."""

    __tablename__ = "lecturers"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), unique=True, index=True, nullable=False
    )

    employee_id: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    department: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(100), default="Lecturer", nullable=False)
    office_location: Mapped[str | None] = mapped_column(String(200), nullable=True)
    specialization: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    tenure_status: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    hire_date: Mapped[date] = mapped_column(Date, nullable=False)
    max_course_load: Mapped[int] = mapped_column(Integer, default=4, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class StaffModel(Base):
    """Staff-specific data model."""

    __tablename__ = "staff"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), unique=True, index=True, nullable=False
    )

    employee_id: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    department: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    job_title: Mapped[str] = mapped_column(String(100), nullable=False)
    office_location: Mapped[str | None] = mapped_column(String(200), nullable=True)
    hire_date: Mapped[date] = mapped_column(Date, nullable=False)
    clearance_level: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

