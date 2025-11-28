"""
Grade Management API Endpoints

CRUD operations for grade assignment - fully functional.
Admins and Lecturers can assign grades to students.
"""

import base64
from datetime import datetime
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.academic_service.models import EnrollmentModel, GradeModel, SectionModel
from shared.database import get_db
from shared.security.encryption import EncryptionService

router = APIRouter(prefix="/grades", tags=["grades"])
logger = structlog.get_logger(__name__)

# Get encryption service instance
_encryption_service: EncryptionService | None = None

def get_encryption_service_instance() -> EncryptionService:
    """Get or create encryption service instance."""
    global _encryption_service
    if _encryption_service is None:
        import os
        # Get encryption key from environment or generate new one
        key_str = os.getenv("ENCRYPTION_KEY")
        key = base64.b64decode(key_str) if key_str else EncryptionService.generate_key()
        _encryption_service = EncryptionService(master_key=key)
    return _encryption_service


class CreateGradeRequest(BaseModel):
    """Create grade request."""

    student_id: UUID
    section_id: UUID
    assessment_id: UUID = Field(default_factory=lambda: UUID('00000000-0000-0000-0000-000000000000'))
    points_earned: float = Field(..., ge=0.0)
    total_points: float = Field(..., gt=0.0)
    feedback: str | None = Field(None, max_length=5000)
    is_late: bool = Field(default=False)
    late_days: int = Field(default=0, ge=0)


class GradeResponse(BaseModel):
    """Grade response."""

    id: UUID
    student_id: UUID
    section_id: UUID
    assessment_id: UUID
    points_earned: float
    total_points: float
    percentage: float
    letter_grade: str
    graded_by: UUID
    graded_at: datetime
    feedback: str | None
    is_late: bool
    version: int


async def get_current_user_id(authorization: str | None = Header(None)) -> UUID | None:
    """Extract user ID from JWT token."""
    if not authorization:
        return None

    try:
        from jose import jwt

        from shared.config import settings

        token = authorization.replace("Bearer ", "").strip()
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        return UUID(payload.get("sub"))

    except Exception as e:
        logger.warning("JWT validation failed", error=str(e))
        return None


async def verify_grader(
    authorization: str | None = Header(None),
    section_id: UUID | None = None,
    db: AsyncSession = Depends(get_db),
) -> UUID:
    """
    Verify user can grade (admin or section instructor).

    Args:
        authorization: JWT token
        section_id: Section ID to check instructor
        db: Database session

    Returns:
        User ID of grader
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    try:
        import httpx
        from jose import jwt

        from shared.config import settings

        token = authorization.replace("Bearer ", "").strip()
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        user_id = UUID(payload.get("sub"))

        # Verify user type via user service
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"http://localhost:{settings.user_service_port}/api/v1/users/me",
                headers={"Authorization": authorization},
            )
            response.raise_for_status()
            user_data = response.json()
            user_type = user_data.get("user_type")

            # Admins can always grade
            if user_type == "admin":
                return user_id

            # Lecturers can grade their own sections
            if user_type == "lecturer" and section_id:
                section_result = await db.execute(
                    select(SectionModel).where(SectionModel.id == section_id)
                )
                section = section_result.scalar_one_or_none()
                if section and section.instructor_id == user_id:
                    return user_id

            raise HTTPException(
                status_code=403,
                detail="Access denied - Admin or section instructor required"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Grader verification failed", error=str(e))
        raise HTTPException(status_code=401, detail="Invalid authentication token")


@router.post("", response_model=GradeResponse, status_code=status.HTTP_201_CREATED)
async def create_grade(
    request: CreateGradeRequest,
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
) -> GradeResponse:
    """
    Create a new grade for a student.

    This updates the enrollment's current_grade_percentage and current_letter_grade.
    Grades are immutable - new versions are created for regrades.

    Args:
        request: Grade creation request
        authorization: JWT token
        db: Database session

    Returns:
        GradeResponse: Created grade
    """
    # Verify grader permissions
    grader_id = await verify_grader(authorization, request.section_id, db)

    # Verify enrollment exists
    enrollment_result = await db.execute(
        select(EnrollmentModel).where(
            EnrollmentModel.student_id == request.student_id,
            EnrollmentModel.section_id == request.section_id,
        )
    )
    enrollment = enrollment_result.scalar_one_or_none()

    if not enrollment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Enrollment not found"
        )

    # Calculate percentage and letter grade
    percentage = (request.points_earned / request.total_points) * 100.0 if request.total_points > 0 else 0.0

    # Convert percentage to letter grade
    if percentage >= 93:
        letter_grade = "A"
    elif percentage >= 90:
        letter_grade = "A-"
    elif percentage >= 87:
        letter_grade = "B+"
    elif percentage >= 83:
        letter_grade = "B"
    elif percentage >= 80:
        letter_grade = "B-"
    elif percentage >= 77:
        letter_grade = "C+"
    elif percentage >= 73:
        letter_grade = "C"
    elif percentage >= 70:
        letter_grade = "C-"
    elif percentage >= 67:
        letter_grade = "D+"
    elif percentage >= 63:
        letter_grade = "D"
    elif percentage >= 60:
        letter_grade = "D-"
    else:
        letter_grade = "F"

    # Encrypt sensitive grade data
    encryption_service = get_encryption_service_instance()
    encrypted_points = encryption_service.encrypt_field(request.points_earned)
    encrypted_total = encryption_service.encrypt_field(request.total_points)
    encrypted_percentage = encryption_service.encrypt_field(percentage)
    encrypted_feedback = encryption_service.encrypt_field(request.feedback) if request.feedback else None

    # Create grade with encrypted fields
    grade = GradeModel(
        student_id=request.student_id,
        section_id=request.section_id,
        assessment_id=request.assessment_id,
        points_earned=encrypted_points,  # Encrypted
        total_points=encrypted_total,  # Encrypted
        percentage=encrypted_percentage,  # Encrypted
        letter_grade=letter_grade,
        graded_by=grader_id,
        graded_at=datetime.utcnow(),
        feedback=encrypted_feedback,  # Encrypted
        is_late=request.is_late,
        version=1,
    )

    db.add(grade)

    # Update enrollment with current grade
    enrollment.current_grade_percentage = percentage
    enrollment.current_letter_grade = letter_grade
    enrollment.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(grade)

    logger.info(
        "Grade created",
        grade_id=str(grade.id),
        student_id=str(request.student_id),
        section_id=str(request.section_id),
    )

    # Decrypt grade data for response
    encryption_service = get_encryption_service_instance()
    decrypted_points = encryption_service.decrypt_field(grade.points_earned)
    decrypted_total = encryption_service.decrypt_field(grade.total_points)
    decrypted_percentage = encryption_service.decrypt_field(grade.percentage)
    decrypted_feedback = encryption_service.decrypt_field(grade.feedback) if grade.feedback else None

    return GradeResponse(
        id=grade.id,
        student_id=grade.student_id,
        section_id=grade.section_id,
        assessment_id=grade.assessment_id,
        points_earned=float(decrypted_points),
        total_points=float(decrypted_total),
        percentage=float(decrypted_percentage),
        letter_grade=grade.letter_grade,
        graded_by=grade.graded_by,
        graded_at=grade.graded_at,
        feedback=decrypted_feedback,
        is_late=grade.is_late,
        version=grade.version,
    )


@router.get("", response_model=list[GradeResponse])
async def list_grades(
    student_id: UUID | None = None,
    section_id: UUID | None = None,
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
) -> list[GradeResponse]:
    """
    List grades with filtering.

    Args:
        student_id: Filter by student
        section_id: Filter by section
        authorization: JWT token
        db: Database session

    Returns:
        List of grades
    """
    query = select(GradeModel)

    if student_id:
        query = query.where(GradeModel.student_id == student_id)
    if section_id:
        query = query.where(GradeModel.section_id == section_id)

    # Get latest version of each grade (for regrades)
    query = query.order_by(GradeModel.student_id, GradeModel.section_id, GradeModel.version.desc())

    result = await db.execute(query)
    grades = result.scalars().all()

    # Filter to latest version only
    seen = set()
    latest_grades = []
    for grade in grades:
        key = (grade.student_id, grade.section_id, grade.assessment_id)
        if key not in seen:
            seen.add(key)
            latest_grades.append(grade)

    # Decrypt grades for response
    encryption_service = get_encryption_service_instance()
    decrypted_grades = []
    for grade in latest_grades:
        decrypted_points = encryption_service.decrypt_field(grade.points_earned)
        decrypted_total = encryption_service.decrypt_field(grade.total_points)
        decrypted_percentage = encryption_service.decrypt_field(grade.percentage)
        decrypted_feedback = encryption_service.decrypt_field(grade.feedback) if grade.feedback else None

        decrypted_grades.append(
            GradeResponse(
                id=grade.id,
                student_id=grade.student_id,
                section_id=grade.section_id,
                assessment_id=grade.assessment_id,
                points_earned=float(decrypted_points),
                total_points=float(decrypted_total),
                percentage=float(decrypted_percentage),
                letter_grade=grade.letter_grade,
                graded_by=grade.graded_by,
                graded_at=grade.graded_at,
                feedback=decrypted_feedback,
                is_late=grade.is_late,
                version=grade.version,
            )
        )

    return decrypted_grades

