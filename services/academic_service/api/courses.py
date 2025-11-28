"""
Course Management API Endpoints

CRUD operations for courses - fully functional, no mocks.
"""

from datetime import datetime
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.academic_service.models import CourseModel
from shared.database import get_db

logger = structlog.get_logger(__name__)

router = APIRouter()


# Request/Response Models
class CreateCourseRequest(BaseModel):
    """Create course request."""

    course_code: str = Field(..., min_length=3, max_length=20)
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1)
    credits: int = Field(..., ge=1, le=12)
    level: str = Field(..., description="undergraduate, graduate, doctoral")
    department: str = Field(..., max_length=100)
    prerequisites: list[str] = Field(default_factory=list)
    corequisites: list[str] = Field(default_factory=list)


class UpdateCourseRequest(BaseModel):
    """Update course request."""

    title: str | None = None
    description: str | None = None
    prerequisites: list[str] | None = None
    max_enrollment_default: int | None = None


class CourseResponse(BaseModel):
    """Course response."""

    id: UUID
    course_code: str
    title: str
    description: str
    credits: int
    level: str
    department: str
    prerequisites: list[str]
    corequisites: list[str]
    learning_outcomes: list[str]
    is_lab_required: bool
    max_enrollment_default: int
    created_at: datetime
    status: str


@router.post("", response_model=CourseResponse, status_code=status.HTTP_201_CREATED)
async def create_course(
    request: CreateCourseRequest, db: AsyncSession = Depends(get_db)
) -> CourseResponse:
    """
    Create a new course.

    Args:
        request: Course creation request
        db: Database session

    Returns:
        CourseResponse: Created course

    Raises:
        HTTPException: If course code already exists
    """
    logger.info("Creating course", course_code=request.course_code)

    # Check if course code already exists
    result = await db.execute(
        select(CourseModel).where(CourseModel.course_code == request.course_code.upper())
    )
    existing = result.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Course with code {request.course_code} already exists",
        )

    # Create course
    course = CourseModel(
        course_code=request.course_code.upper(),
        title=request.title,
        description=request.description,
        credits=request.credits,
        level=request.level,
        department=request.department,
        prerequisites=request.prerequisites,
        corequisites=request.corequisites,
    )

    db.add(course)
    await db.commit()
    await db.refresh(course)

    logger.info("Course created", course_id=str(course.id), course_code=course.course_code)

    return CourseResponse(
        id=course.id,
        course_code=course.course_code,
        title=course.title,
        description=course.description,
        credits=course.credits,
        level=course.level,
        department=course.department,
        prerequisites=course.prerequisites,
        corequisites=course.corequisites,
        learning_outcomes=course.learning_outcomes,
        is_lab_required=course.is_lab_required,
        max_enrollment_default=course.max_enrollment_default,
        created_at=course.created_at,
        status=course.status,
    )


@router.get("", response_model=list[CourseResponse])
async def list_courses(
    department: str | None = Query(None),
    level: str | None = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> list[CourseResponse]:
    """
    List courses with optional filtering.

    Args:
        department: Filter by department
        level: Filter by level
        skip: Pagination offset
        limit: Page size
        db: Database session

    Returns:
        List of courses
    """
    query = select(CourseModel).where(CourseModel.status == "active")

    if department:
        query = query.where(CourseModel.department == department)

    if level:
        query = query.where(CourseModel.level == level)

    query = query.offset(skip).limit(limit)

    result = await db.execute(query)
    courses = result.scalars().all()

    return [
        CourseResponse(
            id=course.id,
            course_code=course.course_code,
            title=course.title,
            description=course.description,
            credits=course.credits,
            level=course.level,
            department=course.department,
            prerequisites=course.prerequisites,
            corequisites=course.corequisites,
            learning_outcomes=course.learning_outcomes,
            is_lab_required=course.is_lab_required,
            max_enrollment_default=course.max_enrollment_default,
            created_at=course.created_at,
            status=course.status,
        )
        for course in courses
    ]


@router.get("/{course_id}", response_model=CourseResponse)
async def get_course(course_id: UUID, db: AsyncSession = Depends(get_db)) -> CourseResponse:
    """
    Get course by ID.

    Args:
        course_id: Course UUID
        db: Database session

    Returns:
        CourseResponse: Course details

    Raises:
        HTTPException: If course not found
    """
    result = await db.execute(select(CourseModel).where(CourseModel.id == course_id))
    course = result.scalar_one_or_none()

    if course is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")

    return CourseResponse(
        id=course.id,
        course_code=course.course_code,
        title=course.title,
        description=course.description,
        credits=course.credits,
        level=course.level,
        department=course.department,
        prerequisites=course.prerequisites,
        corequisites=course.corequisites,
        learning_outcomes=course.learning_outcomes,
        is_lab_required=course.is_lab_required,
        max_enrollment_default=course.max_enrollment_default,
        created_at=course.created_at,
        status=course.status,
    )


@router.put("/{course_id}", response_model=CourseResponse)
async def update_course(
    course_id: UUID,
    request: UpdateCourseRequest,
    db: AsyncSession = Depends(get_db),
) -> CourseResponse:
    """
    Update course information.

    Args:
        course_id: Course UUID
        request: Update request
        db: Database session

    Returns:
        CourseResponse: Updated course

    Raises:
        HTTPException: If course not found
    """
    result = await db.execute(select(CourseModel).where(CourseModel.id == course_id))
    course = result.scalar_one_or_none()

    if course is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")

    # Update fields
    if request.title is not None:
        course.title = request.title
    if request.description is not None:
        course.description = request.description
    if request.prerequisites is not None:
        course.prerequisites = request.prerequisites
    if request.max_enrollment_default is not None:
        course.max_enrollment_default = request.max_enrollment_default

    course.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(course)

    logger.info("Course updated", course_id=str(course_id))

    return CourseResponse(
        id=course.id,
        course_code=course.course_code,
        title=course.title,
        description=course.description,
        credits=course.credits,
        level=course.level,
        department=course.department,
        prerequisites=course.prerequisites,
        corequisites=course.corequisites,
        learning_outcomes=course.learning_outcomes,
        is_lab_required=course.is_lab_required,
        max_enrollment_default=course.max_enrollment_default,
        created_at=course.created_at,
        status=course.status,
    )


@router.delete("/{course_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_course(
    course_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Delete a course (soft delete by setting status to inactive).

    Args:
        course_id: Course UUID
        db: Database session

    Raises:
        HTTPException: If course not found
    """
    result = await db.execute(select(CourseModel).where(CourseModel.id == course_id))
    course = result.scalar_one_or_none()

    if course is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")

    # Soft delete by setting status to inactive
    course.status = "inactive"
    await db.commit()

    logger.info("Course deleted (soft delete)", course_id=str(course_id))

    return

