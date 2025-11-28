"""
Section Management API Endpoints

CRUD operations for course sections - fully functional.
"""

from datetime import date, datetime
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.academic_service.models import CourseModel, SectionModel
from shared.database import get_db

logger = structlog.get_logger(__name__)

router = APIRouter()


class CreateSectionRequest(BaseModel):
    """Create section request."""

    course_id: UUID
    section_number: str = Field(..., min_length=1, max_length=10)
    semester: str = Field(..., description="e.g., 'Fall 2024'")
    instructor_id: UUID
    schedule_days: list[str] = Field(..., min_items=1)
    start_time: str = Field(..., description="HH:MM format")
    end_time: str = Field(..., description="HH:MM format")
    room_id: UUID | None = None
    max_enrollment: int = Field(default=30, ge=1, le=500)
    max_waitlist: int = Field(default=10, ge=0)
    start_date: date
    end_date: date
    add_drop_deadline: date
    withdrawal_deadline: date


class SectionResponse(BaseModel):
    """Section response."""

    id: UUID
    course_id: UUID
    course_code: str
    course_title: str
    section_number: str
    semester: str
    instructor_id: UUID
    schedule_days: list[str]
    start_time: str
    end_time: str
    room_id: UUID | None
    max_enrollment: int
    current_enrollment: int
    waitlist_size: int
    max_waitlist: int
    is_full: bool
    has_waitlist_space: bool
    start_date: date
    end_date: date
    add_drop_deadline: date
    withdrawal_deadline: date
    created_at: datetime


@router.post("", response_model=SectionResponse, status_code=status.HTTP_201_CREATED)
async def create_section(
    request: CreateSectionRequest, db: AsyncSession = Depends(get_db)
) -> SectionResponse:
    """
    Create a new course section.

    Args:
        request: Section creation request
        db: Database session

    Returns:
        SectionResponse: Created section
    """
    logger.info(
        "Creating section",
        course_id=str(request.course_id),
        section_number=request.section_number,
    )

    # Verify course exists
    course_result = await db.execute(
        select(CourseModel).where(CourseModel.id == request.course_id)
    )
    course = course_result.scalar_one_or_none()

    if course is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")

    # Create section
    section = SectionModel(
        course_id=request.course_id,
        section_number=request.section_number,
        semester=request.semester,
        instructor_id=request.instructor_id,
        schedule_days=request.schedule_days,
        start_time=request.start_time,
        end_time=request.end_time,
        room_id=request.room_id,
        max_enrollment=request.max_enrollment,
        max_waitlist=request.max_waitlist,
        start_date=request.start_date,
        end_date=request.end_date,
        add_drop_deadline=request.add_drop_deadline,
        withdrawal_deadline=request.withdrawal_deadline,
    )

    db.add(section)
    await db.commit()
    await db.refresh(section)

    logger.info("Section created", section_id=str(section.id))

    return SectionResponse(
        id=section.id,
        course_id=section.course_id,
        course_code=course.course_code,
        course_title=course.title,
        section_number=section.section_number,
        semester=section.semester,
        instructor_id=section.instructor_id,
        schedule_days=section.schedule_days,
        start_time=section.start_time,
        end_time=section.end_time,
        room_id=section.room_id,
        max_enrollment=section.max_enrollment,
        current_enrollment=section.current_enrollment,
        waitlist_size=section.waitlist_size,
        max_waitlist=section.max_waitlist,
        is_full=section.current_enrollment >= section.max_enrollment,
        has_waitlist_space=section.waitlist_size < section.max_waitlist,
        start_date=section.start_date,
        end_date=section.end_date,
        add_drop_deadline=section.add_drop_deadline,
        withdrawal_deadline=section.withdrawal_deadline,
        created_at=section.created_at,
    )


@router.get("", response_model=list[SectionResponse])
async def list_sections(
    course_code: str | None = Query(None),
    semester: str | None = Query(None),
    available_only: bool = Query(False),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> list[SectionResponse]:
    """
    List sections with filtering.

    Args:
        course_code: Filter by course code
        semester: Filter by semester
        available_only: Show only sections with space
        skip: Pagination offset
        limit: Page size
        db: Database session

    Returns:
        List of sections
    """
    try:
        logger.info(
            "Listing sections",
            course_code=course_code,
            semester=semester,
            available_only=available_only,
            skip=skip,
            limit=limit,
        )

        # Join sections with courses
        query = select(SectionModel, CourseModel).join(
            CourseModel, SectionModel.course_id == CourseModel.id
        )

        if course_code:
            query = query.where(CourseModel.course_code == course_code.upper())

        if semester:
            query = query.where(SectionModel.semester == semester)

        if available_only:
            query = query.where(SectionModel.current_enrollment < SectionModel.max_enrollment)

        query = query.offset(skip).limit(limit)

        result = await db.execute(query)
        section_course_pairs = result.all()

        sections_response = []
        for section, course in section_course_pairs:
            sections_response.append(
                SectionResponse(
                    id=section.id,
                    course_id=section.course_id,
                    course_code=course.course_code,
                    course_title=course.title,
                    section_number=section.section_number,
                    semester=section.semester,
                    instructor_id=section.instructor_id,
                    schedule_days=section.schedule_days,
                    start_time=section.start_time,
                    end_time=section.end_time,
                    room_id=section.room_id,
                    max_enrollment=section.max_enrollment,
                    current_enrollment=section.current_enrollment,
                    waitlist_size=section.waitlist_size,
                    max_waitlist=section.max_waitlist,
                    is_full=section.current_enrollment >= section.max_enrollment,
                    has_waitlist_space=section.waitlist_size < section.max_waitlist,
                    start_date=section.start_date,
                    end_date=section.end_date,
                    add_drop_deadline=section.add_drop_deadline,
                    withdrawal_deadline=section.withdrawal_deadline,
                    created_at=section.created_at,
                )
            )

        logger.info("Sections listed", count=len(sections_response))
        return sections_response

    except Exception as e:
        logger.error(
            "Error listing sections",
            error=str(e),
            error_type=type(e).__name__,
            course_code=course_code,
            semester=semester,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sections: {str(e)}"
        )


@router.get("/{section_id}", response_model=SectionResponse)
async def get_section(section_id: UUID, db: AsyncSession = Depends(get_db)) -> SectionResponse:
    """
    Get section by ID.

    Args:
        section_id: Section UUID
        db: Database session

    Returns:
        SectionResponse: Section details
    """
    result = await db.execute(
        select(SectionModel, CourseModel)
        .join(CourseModel, SectionModel.course_id == CourseModel.id)
        .where(SectionModel.id == section_id)
    )
    row = result.one_or_none()

    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Section not found")

    section, course = row

    return SectionResponse(
        id=section.id,
        course_id=section.course_id,
        course_code=course.course_code,
        course_title=course.title,
        section_number=section.section_number,
        semester=section.semester,
        instructor_id=section.instructor_id,
        schedule_days=section.schedule_days,
        start_time=section.start_time,
        end_time=section.end_time,
        room_id=section.room_id,
        max_enrollment=section.max_enrollment,
        current_enrollment=section.current_enrollment,
        waitlist_size=section.waitlist_size,
        max_waitlist=section.max_waitlist,
        is_full=section.current_enrollment >= section.max_enrollment,
        has_waitlist_space=section.waitlist_size < section.max_waitlist,
        start_date=section.start_date,
        end_date=section.end_date,
        add_drop_deadline=section.add_drop_deadline,
        withdrawal_deadline=section.withdrawal_deadline,
        created_at=section.created_at,
    )

