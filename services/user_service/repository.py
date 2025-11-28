"""
User Service Repository

Database access layer for user operations using repository pattern.
"""

from datetime import date, datetime
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.user_service.auth_utils import password_hasher
from services.user_service.models import StudentModel, UserModel

logger = structlog.get_logger(__name__)


class UserRepository:
    """Repository for user data access."""

    def __init__(self, session: AsyncSession):
        """
        Initialize repository.

        Args:
            session: Database session
        """
        self.session = session

    async def create_user(
        self,
        email: str,
        password: str,
        first_name: str,
        last_name: str,
        user_type: str,
        **kwargs,
    ) -> UserModel:
        """
        Create a new user.

        Args:
            email: User email
            password: Plain text password (will be hashed)
            first_name: First name
            last_name: Last name
            user_type: Type of user (student, lecturer, staff, etc.)
            **kwargs: Additional fields

        Returns:
            UserModel: Created user
        """
        # Hash password
        password_hash = password_hasher.hash_password(password)

        # Create user
        user = UserModel(
            email=email.lower(),
            password_hash=password_hash,
            first_name=first_name,
            last_name=last_name,
            user_type=user_type,
            **kwargs,
        )

        self.session.add(user)
        await self.session.flush()

        logger.info("User created", user_id=str(user.id), email=user.email, type=user_type)

        return user

    async def get_user_by_id(self, user_id: UUID) -> UserModel | None:
        """
        Get user by ID.

        Args:
            user_id: User UUID

        Returns:
            UserModel or None
        """
        result = await self.session.execute(select(UserModel).where(UserModel.id == user_id))
        return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str) -> UserModel | None:
        """
        Get user by email.

        Args:
            email: User email

        Returns:
            UserModel or None
        """
        result = await self.session.execute(
            select(UserModel).where(UserModel.email == email.lower())
        )
        return result.scalar_one_or_none()

    async def verify_password(self, email: str, password: str) -> UserModel | None:
        """
        Verify user credentials.

        Args:
            email: User email
            password: Plain text password

        Returns:
            UserModel if credentials are valid, None otherwise
        """
        user = await self.get_user_by_email(email)

        if user is None:
            logger.warning("Login attempt for non-existent user", email=email)
            return None

        if not user.is_active:
            logger.warning("Login attempt for inactive user", email=email)
            return None

        if not password_hasher.verify_password(password, user.password_hash):
            logger.warning("Login attempt with invalid password", email=email)
            return None

        # Update last login
        user.last_login_at = datetime.utcnow()
        await self.session.flush()

        logger.info("User authenticated", user_id=str(user.id), email=email)

        return user

    async def update_user(self, user_id: UUID, **updates) -> UserModel | None:
        """
        Update user fields.

        Args:
            user_id: User UUID
            **updates: Fields to update

        Returns:
            Updated UserModel or None if not found
        """
        user = await self.get_user_by_id(user_id)

        if user is None:
            return None

        for key, value in updates.items():
            if hasattr(user, key):
                setattr(user, key, value)

        user.updated_at = datetime.utcnow()
        await self.session.flush()

        logger.info("User updated", user_id=str(user_id), fields=list(updates.keys()))

        return user


class StudentRepository:
    """Repository for student-specific data."""

    def __init__(self, session: AsyncSession):
        """Initialize repository."""
        self.session = session

    async def create_student(
        self, user_id: UUID, student_id: str, enrollment_date: date, **kwargs
    ) -> StudentModel:
        """Create student record."""
        student = StudentModel(
            user_id=user_id,
            student_id=student_id,
            enrollment_date=enrollment_date,
            **kwargs,
        )

        self.session.add(student)
        await self.session.flush()

        logger.info("Student record created", user_id=str(user_id), student_id=student_id)

        return student

    async def get_student_by_user_id(self, user_id: UUID) -> StudentModel | None:
        """Get student by user ID."""
        result = await self.session.execute(
            select(StudentModel).where(StudentModel.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_student_by_student_id(self, student_id: str) -> StudentModel | None:
        """Get student by student ID."""
        result = await self.session.execute(
            select(StudentModel).where(StudentModel.student_id == student_id)
        )
        return result.scalar_one_or_none()

    async def update_gpa(self, student_id: UUID, new_gpa: float) -> StudentModel | None:
        """Update student GPA."""
        result = await self.session.execute(
            select(StudentModel).where(StudentModel.id == student_id)
        )
        student = result.scalar_one_or_none()

        if student:
            student.gpa = new_gpa
            student.updated_at = datetime.utcnow()
            await self.session.flush()
            logger.info("Student GPA updated", student_id=str(student_id), new_gpa=new_gpa)

        return student

