"""
GDPR Data Erasure Service

Implements safe deletion and pseudonymization of student data while preserving
analytics integrity. Supports both full deletion and pseudonymization modes.
"""

import hashlib
from typing import Any
from uuid import UUID, uuid4

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.events.security_events import DataErasureCompletedEvent, DataErasureRequestEvent
from shared.events.stream import get_event_stream_manager
from shared.security.audit import AuditAction, AuditLogEntry

logger = structlog.get_logger(__name__)


class GDPRDataErasureService:
    """
    Service for GDPR-compliant data erasure and pseudonymization.

    Supports:
    - Full deletion (removes all personal data)
    - Pseudonymization (replaces PII with anonymized identifiers, preserves analytics)
    """

    def __init__(self, db: AsyncSession):
        """
        Initialize GDPR erasure service.

        Args:
            db: Database session
        """
        self.db = db
        self.event_stream = get_event_stream_manager().get_or_create_stream("security_events")

    async def request_erasure(
        self,
        data_subject_id: UUID,
        requested_by: UUID,
        scope: str = "pseudonymize",
        reason: str = "GDPR Right to be Forgotten",
    ) -> dict[str, Any]:
        """
        Request data erasure for a student.

        Args:
            data_subject_id: Student/user ID to erase
            requested_by: User making the request
            scope: "delete" (full deletion) or "pseudonymize" (anonymize, preserve analytics)
            reason: Reason for erasure

        Returns:
            Dictionary with erasure request details
        """
        logger.info(
            "GDPR erasure requested",
            data_subject_id=str(data_subject_id),
            requested_by=str(requested_by),
            scope=scope,
        )

        # Emit event
        event = DataErasureRequestEvent(
            data_subject_id=data_subject_id,
            requested_by=requested_by,
            scope=scope,
            reason=reason,
        )
        await self.event_stream.publish(event)

        # Process erasure
        if scope == "delete":
            result = await self._full_deletion(data_subject_id, requested_by)
        else:
            result = await self._pseudonymize(data_subject_id, requested_by)

        # Emit completion event
        completion_event = DataErasureCompletedEvent(
            data_subject_id=data_subject_id,
            completed_by=requested_by,
            records_affected=result.get("records_affected", 0),
            pseudonymized=(scope == "pseudonymize"),
        )
        await self.event_stream.publish(completion_event)

        return result

    async def _pseudonymize(self, student_id: UUID, requested_by: UUID) -> dict[str, Any]:
        """
        Pseudonymize student data (preserve analytics integrity).

        Replaces PII with anonymized identifiers while maintaining:
        - Referential integrity (foreign keys)
        - Analytics data (grades, enrollments aggregated)
        - System functionality

        Args:
            student_id: Student ID to pseudonymize
            requested_by: User making the request

        Returns:
            Dictionary with pseudonymization results
        """
        from services.user_service.models import StudentModel, UserModel

        records_affected = 0

        # Pseudonymize User record
        user_result = await self.db.execute(
            select(UserModel).where(UserModel.id == student_id)
        )
        user = user_result.scalar_one_or_none()

        if user:
            # Generate pseudonymized identifiers
            pseudonym_id = self._generate_pseudonym(student_id)

            # Update user with pseudonymized data
            user.email = f"user-{pseudonym_id}@pseudonymized.local"
            user.first_name = "User"
            user.last_name = pseudonym_id[:8]
            user.middle_name = None
            user.is_pseudonymized = True

            # Clear sensitive fields
            user.phone_number = None
            user.date_of_birth = None

            records_affected += 1

        # Pseudonymize Student record
        student_result = await self.db.execute(
            select(StudentModel).where(StudentModel.user_id == student_id)
        )
        student = student_result.scalar_one_or_none()

        if student:
            # Preserve academic data but remove PII
            student.student_number = f"STU-{pseudonym_id[:8]}"
            # Keep enrollment, grades, etc. for analytics
            records_affected += 1

        await self.db.commit()

        # Create audit log entry
        AuditLogEntry.create(
            action=AuditAction.UPDATE,
            resource_type="user",
            resource_id=student_id,
            description=f"GDPR pseudonymization completed for user {student_id}",
            actor_id=requested_by,
            metadata={
                "scope": "pseudonymize",
                "records_affected": records_affected,
                "pseudonym_id": pseudonym_id,
            },
        )

        logger.info(
            "GDPR pseudonymization completed",
            student_id=str(student_id),
            records_affected=records_affected,
        )

        return {
            "status": "pseudonymized",
            "student_id": str(student_id),
            "pseudonym_id": pseudonym_id,
            "records_affected": records_affected,
            "analytics_preserved": True,
        }

    async def _full_deletion(self, student_id: UUID, requested_by: UUID) -> dict[str, Any]:
        """
        Perform full deletion of student data.

        WARNING: This removes all personal data. Analytics data may be aggregated
        and preserved for statistical purposes (anonymized).

        Args:
            student_id: Student ID to delete
            requested_by: User making the request

        Returns:
            Dictionary with deletion results
        """
        from services.academic_service.models import EnrollmentModel, GradeModel
        from services.user_service.models import StudentModel, UserModel

        records_affected = 0

        # Delete grades (or anonymize for analytics)
        grade_result = await self.db.execute(
            select(GradeModel).where(GradeModel.student_id == student_id)
        )
        grades = grade_result.scalars().all()
        for grade in grades:
            # Anonymize grade (preserve for analytics but remove student link)
            grade.student_id = uuid4()  # Replace with anonymized ID
            records_affected += 1

        # Delete enrollments (or anonymize)
        enrollment_result = await self.db.execute(
            select(EnrollmentModel).where(EnrollmentModel.student_id == student_id)
        )
        enrollments = enrollment_result.scalars().all()
        for enrollment in enrollments:
            enrollment.student_id = uuid4()  # Anonymize
            records_affected += 1

        # Delete student record
        student_result = await self.db.execute(
            select(StudentModel).where(StudentModel.user_id == student_id)
        )
        student = student_result.scalar_one_or_none()
        if student:
            await self.db.delete(student)
            records_affected += 1

        # Delete user record
        user_result = await self.db.execute(
            select(UserModel).where(UserModel.id == student_id)
        )
        user = user_result.scalar_one_or_none()
        if user:
            await self.db.delete(user)
            records_affected += 1

        await self.db.commit()

        # Create audit log entry
        AuditLogEntry.create(
            action=AuditAction.DELETE,
            resource_type="user",
            resource_id=student_id,
            description=f"GDPR full deletion completed for user {student_id}",
            actor_id=requested_by,
            metadata={
                "scope": "delete",
                "records_affected": records_affected,
            },
        )

        logger.info(
            "GDPR full deletion completed",
            student_id=str(student_id),
            records_affected=records_affected,
        )

        return {
            "status": "deleted",
            "student_id": str(student_id),
            "records_affected": records_affected,
            "analytics_preserved": True,  # Aggregated data preserved
        }

    def _generate_pseudonym(self, user_id: UUID) -> str:
        """
        Generate a deterministic pseudonym for a user ID.

        Uses SHA-256 hash to create consistent pseudonym that can be
        used for analytics while maintaining anonymity.

        Args:
            user_id: User UUID

        Returns:
            Pseudonym string (hex digest)
        """
        # Use salt + user_id for pseudonym generation
        salt = b"gdpr_pseudonym_salt_v1"  # In production, use secure salt from config
        hash_input = salt + user_id.bytes
        return hashlib.sha256(hash_input).hexdigest()[:16]

    async def verify_erasure(self, student_id: UUID) -> dict[str, Any]:
        """
        Verify that data erasure was completed successfully.

        Args:
            student_id: Student ID to verify

        Returns:
            Dictionary with verification results
        """
        from services.user_service.models import UserModel

        user_result = await self.db.execute(
            select(UserModel).where(UserModel.id == student_id)
        )
        user = user_result.scalar_one_or_none()

        if not user:
            return {
                "status": "deleted",
                "verified": True,
                "message": "User record not found - deletion confirmed",
            }

        if user.is_pseudonymized:
            return {
                "status": "pseudonymized",
                "verified": True,
                "message": "User data has been pseudonymized",
                "pseudonymized_email": user.email,
            }

        return {
            "status": "not_erased",
            "verified": False,
            "message": "User data has not been erased",
        }

