"""
Lecturer-specific endpoints for course and student management.
"""

from typing import Any
from uuid import UUID

import httpx
import structlog
from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.academic_service.models import CourseModel, EnrollmentModel, SectionModel
from shared.config import settings
from shared.database import get_db

router = APIRouter(prefix="/lecturer", tags=["lecturer"])
logger = structlog.get_logger(__name__)


async def get_current_lecturer_id(authorization: str | None = Header(None)) -> UUID:
    """Extract lecturer ID from JWT token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    try:
        from jose import jwt

        from shared.config import settings

        # Extract token
        token = authorization.replace("Bearer ", "")

        # Decode JWT
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        return UUID(payload.get("sub"))

        # Note: user_type is not in JWT payload, so we verify via user service
        # For now, we'll just verify JWT is valid and trust the API Gateway
        # TODO: Add proper lecturer verification via user service API


    except Exception as e:
        logger.error("JWT validation failed", error=str(e))
        raise HTTPException(status_code=401, detail="Invalid authentication token")


@router.get("/sections")
async def get_lecturer_sections(
    db: AsyncSession = Depends(get_db),
    lecturer_id: UUID = Depends(get_current_lecturer_id),
):
    """
    Get all sections taught by the current lecturer.

    Returns:
        List of sections with enrollment information
    """
    try:
        # Query sections taught by this lecturer
        stmt = (
            select(SectionModel, CourseModel)
            .join(CourseModel, SectionModel.course_id == CourseModel.id)
            .where(SectionModel.instructor_id == lecturer_id)
        )

        result = await db.execute(stmt)
        rows = result.all()

        sections = []
        for section_model, course_model in rows:
            sections.append({
                "id": str(section_model.id),
                "course_id": str(course_model.id),
                "course_code": course_model.course_code,
                "course_title": course_model.title,
                "section_number": section_model.section_number,
                "semester": section_model.semester,
                "current_enrollment": section_model.current_enrollment,
                "max_enrollment": section_model.max_enrollment,
                "schedule_days": section_model.schedule_days,
                "start_time": section_model.start_time,
                "end_time": section_model.end_time,
            })

        logger.info(
            "Lecturer sections retrieved",
            lecturer_id=str(lecturer_id),
            count=len(sections),
        )

        return sections

    except Exception as e:
        logger.error("Failed to get lecturer sections", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assessments")
async def get_lecturer_assessments(
    db: AsyncSession = Depends(get_db),
    lecturer_id: UUID = Depends(get_current_lecturer_id),
):
    """
    Get all assessments for courses taught by the lecturer.

    Returns:
        List of assessments with grading status
    """
    # TODO: Implement assessment model and queries
    # For now, return empty list
    return []


@router.get("/students")
async def get_lecturer_students(
    section_id: UUID | None = None,
    db: AsyncSession = Depends(get_db),
    lecturer_id: UUID = Depends(get_current_lecturer_id),
):
    """Get all students enrolled in lecturer's courses.

    Returns rich enrollment information so lecturer UIs can show
    student + course + section details without requiring admin APIs.
    """
    try:
        # Build base query for enrollments joined with sections and courses
        if section_id:
            stmt = (
                select(EnrollmentModel, SectionModel, CourseModel)
                .join(SectionModel, EnrollmentModel.section_id == SectionModel.id)
                .join(CourseModel, SectionModel.course_id == CourseModel.id)
                .where(EnrollmentModel.section_id == section_id)
                .where(EnrollmentModel.enrollment_status == "enrolled")
            )
        else:
            # Get all sections for this lecturer first
            sections_stmt = select(SectionModel).where(
                SectionModel.instructor_id == lecturer_id
            )
            sections_result = await db.execute(sections_stmt)
            section_ids = [s.id for s in sections_result.scalars().all()]

            if not section_ids:
                logger.info("Lecturer has no sections", lecturer_id=str(lecturer_id))
                return []

            stmt = (
                select(EnrollmentModel, SectionModel, CourseModel)
                .join(SectionModel, EnrollmentModel.section_id == SectionModel.id)
                .join(CourseModel, SectionModel.course_id == CourseModel.id)
                .where(EnrollmentModel.section_id.in_(section_ids))
                .where(EnrollmentModel.enrollment_status == "enrolled")
            )

        result = await db.execute(stmt)
        rows = result.all()

        # Fetch real student profiles from User Service (one call per unique student)
        student_profiles: dict[str, dict[str, Any]] = {}
        async with httpx.AsyncClient(timeout=5.0) as client:
            for enrollment, _, _ in rows:
                student_id_str = str(enrollment.student_id)
                if student_id_str in student_profiles:
                    continue

                try:
                    resp = await client.get(
                        f"http://localhost:{settings.user_service_port}/api/v1/users/{student_id_str}"
                    )
                    if resp.status_code == 200:
                        profile = resp.json()
                        student_profiles[student_id_str] = {
                            "name": profile.get("full_name")
                            or f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip(),
                            "email": profile.get("email"),
                        }
                    else:
                        # Fallback synthetic details if user not found / error
                        student_profiles[student_id_str] = {
                            "name": f"Student {student_id_str[:8]}",
                            "email": f"student+{student_id_str[:8]}@example.edu",
                        }
                except Exception as e:
                    logger.warning(
                        "Failed to fetch student profile from user service",
                        student_id=student_id_str,
                        error=str(e),
                    )
                    student_profiles[student_id_str] = {
                        "name": f"Student {student_id_str[:8]}",
                        "email": f"student+{student_id_str[:8]}@example.edu",
                    }

        # Re-iterate rows now that profiles are available
        students: list[dict] = []
        for enrollment, section, course in rows:
            student_id_str = str(enrollment.student_id)
            profile = student_profiles.get(student_id_str, {})

            students.append(
                {
                    "id": str(enrollment.id),
                    "student_id": student_id_str,
                    "student_name": profile.get("name")
                    or f"Student {student_id_str[:8]}",
                    "student_email": profile.get("email")
                    or f"student+{student_id_str[:8]}@example.edu",
                    "section_id": str(section.id),
                    "section_number": section.section_number,
                    "course_code": course.course_code,
                    "course_title": course.title,
                    "semester": section.semester,
                    "enrollment_status": enrollment.enrollment_status,
                    "current_grade_percentage": enrollment.current_grade_percentage,
                    "current_letter_grade": enrollment.current_letter_grade,
                    "attendance_percentage": enrollment.attendance_percentage or 0.0,
                    "enrolled_at": enrollment.enrolled_at.isoformat(),
                }
            )

        logger.info(
            "Lecturer students retrieved",
            lecturer_id=str(lecturer_id),
            count=len(students),
        )

        return students

    except Exception as e:
        logger.error("Failed to get lecturer students", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

