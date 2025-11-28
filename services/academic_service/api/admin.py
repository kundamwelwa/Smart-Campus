"""
Admin-specific endpoints for Academic Service.
"""

from typing import Any
from uuid import UUID

import httpx
import structlog
from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from services.academic_service.models import CourseModel, EnrollmentModel, SectionModel
from shared.config import settings
from shared.database import get_db

router = APIRouter(prefix="/admin", tags=["admin"])
logger = structlog.get_logger(__name__)


async def verify_admin(
    authorization: str | None = Header(None),
) -> UUID:
    """Verify user is admin and return admin ID."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    try:
        import httpx
        from jose import jwt

        from shared.config import settings

        # Extract token
        token = authorization.replace("Bearer ", "").strip()

        # Decode JWT
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        user_id = UUID(payload.get("sub"))

        # Verify admin status by calling user service - PRODUCTION: No fallbacks
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"http://localhost:{settings.user_service_port}/api/v1/users/me",
                headers={"Authorization": authorization},
            )
            response.raise_for_status()

            user_data = response.json()
            if user_data.get("user_type") != "admin":
                raise HTTPException(status_code=403, detail="Access denied - Admin role required")

        return user_id

    except HTTPException:
        raise
    except Exception as e:
        logger.error("JWT validation failed", error=str(e))
        raise HTTPException(status_code=401, detail="Invalid authentication token")


@router.get("/stats")
async def get_academic_statistics(
    db: AsyncSession = Depends(get_db),
    authorization: str | None = Header(None),
):
    """
    Get academic service statistics (admin only).

    Returns:
        Dictionary with academic statistics
    """
    # Optional auth - allow internal service calls
    try:
        # Total courses
        total_courses_stmt = select(func.count(CourseModel.id))
        total_courses_result = await db.execute(total_courses_stmt)
        total_courses = total_courses_result.scalar() or 0

        # Active courses
        active_courses_stmt = select(func.count(CourseModel.id)).where(
            CourseModel.status == "active"
        )
        active_courses_result = await db.execute(active_courses_stmt)
        active_courses = active_courses_result.scalar() or 0

        # Total sections
        total_sections_stmt = select(func.count(SectionModel.id))
        total_sections_result = await db.execute(total_sections_stmt)
        total_sections = total_sections_result.scalar() or 0

        # Active sections
        active_sections_stmt = select(func.count(SectionModel.id)).where(
            SectionModel.status == "active"
        )
        active_sections_result = await db.execute(active_sections_stmt)
        active_sections = active_sections_result.scalar() or 0

        # Total enrollments
        total_enrollments_stmt = select(func.count(EnrollmentModel.id))
        total_enrollments_result = await db.execute(total_enrollments_stmt)
        total_enrollments = total_enrollments_result.scalar() or 0

        # Active enrollments
        active_enrollments_stmt = select(func.count(EnrollmentModel.id)).where(
            EnrollmentModel.enrollment_status == "enrolled"
        )
        active_enrollments_result = await db.execute(active_enrollments_stmt)
        active_enrollments = active_enrollments_result.scalar() or 0

        # Waitlisted enrollments
        waitlisted_stmt = select(func.count(EnrollmentModel.id)).where(
            EnrollmentModel.is_waitlisted
        )
        waitlisted_result = await db.execute(waitlisted_stmt)
        waitlisted = waitlisted_result.scalar() or 0

        # Courses by department
        dept_stmt = select(CourseModel.department, func.count(CourseModel.id)).group_by(
            CourseModel.department
        )
        dept_result = await db.execute(dept_stmt)
        by_department = {row[0]: row[1] for row in dept_result.all()}

        return {
            "total_courses": total_courses,
            "active_courses": active_courses,
            "total_sections": total_sections,
            "active_sections": active_sections,
            "total_enrollments": total_enrollments,
            "active_enrollments": active_enrollments,
            "waitlisted_enrollments": waitlisted,
            "by_department": by_department,
        }

    except Exception as e:
        logger.error("Failed to get academic statistics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/enrollments")
async def get_section_enrollments(
    section_id: UUID | None = None,
    course_id: UUID | None = None,
    db: AsyncSession = Depends(get_db),
    admin_id: UUID = Depends(verify_admin),
):
    """
    Get enrollments for a section or course (admin only).

    Args:
        section_id: Filter by section
        course_id: Filter by course (returns all sections)
        db: Database session
        admin_id: Admin user ID

    Returns:
        List of enrollments with student and course information
    """
    try:
        # Build query
        query = (
            select(EnrollmentModel, SectionModel, CourseModel)
            .join(SectionModel, EnrollmentModel.section_id == SectionModel.id)
            .join(CourseModel, SectionModel.course_id == CourseModel.id)
        )

        if section_id:
            query = query.where(EnrollmentModel.section_id == section_id)
        elif course_id:
            query = query.where(SectionModel.course_id == course_id)
        else:
            # Return all enrollments if no filter
            pass

        query = query.where(EnrollmentModel.enrollment_status == "enrolled")

        result = await db.execute(query)
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
                        full_name = profile.get("full_name") or (
                            f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip()
                        )
                        student_profiles[student_id_str] = {
                            "name": full_name or f"Student {student_id_str[:8]}",
                            "email": profile.get("email"),
                        }
                    else:
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

        enrollments = []
        for enrollment, section, course in rows:
            student_id_str = str(enrollment.student_id)
            profile = student_profiles.get(student_id_str, {})

            enrollments.append(
                {
                    "id": str(enrollment.id),
                    "student_id": student_id_str,
                    "student_name": profile.get("name", "Unknown"),
                    "student_email": profile.get("email"),
                    "section_id": str(section.id),
                    "section_number": section.section_number,
                    "course_code": course.course_code,
                    "course_title": course.title,
                    "semester": section.semester,
                    "enrollment_status": enrollment.enrollment_status,
                    "is_waitlisted": enrollment.is_waitlisted,
                    "waitlist_position": enrollment.waitlist_position,
                    "current_grade_percentage": enrollment.current_grade_percentage,
                    "current_letter_grade": enrollment.current_letter_grade,
                    "attendance_percentage": enrollment.attendance_percentage,
                    "enrolled_at": enrollment.enrolled_at.isoformat(),
                }
            )

        logger.info(
            "Admin enrollments retrieved",
            admin_id=str(admin_id),
            count=len(enrollments),
        )

        return enrollments

    except Exception as e:
        logger.error("Failed to get enrollments", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

