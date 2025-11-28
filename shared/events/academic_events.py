"""
Academic Domain Events

Events related to academic operations: courses, enrollment, grades, etc.
"""

from datetime import datetime
from typing import ClassVar
from uuid import UUID

from pydantic import Field

from shared.events.base import DomainEvent


class CourseCreatedEvent(DomainEvent):
    """Event emitted when a new course is created."""

    EVENT_TYPE: ClassVar[str] = "academic.course.created"

    course_code: str = Field(...)
    title: str = Field(...)
    credits: int = Field(...)
    department: str = Field(...)


class CourseUpdatedEvent(DomainEvent):
    """Event emitted when a course is updated."""

    EVENT_TYPE: ClassVar[str] = "academic.course.updated"

    changes: dict[str, tuple[any, any]] = Field(..., description="Changed fields (old, new)")


class SectionCreatedEvent(DomainEvent):
    """Event emitted when a course section is created."""

    EVENT_TYPE: ClassVar[str] = "academic.section.created"

    course_id: UUID = Field(...)
    section_number: str = Field(...)
    semester: str = Field(...)
    instructor_id: UUID = Field(...)
    max_enrollment: int = Field(...)


class SectionScheduledEvent(DomainEvent):
    """Event emitted when a section is scheduled."""

    EVENT_TYPE: ClassVar[str] = "academic.section.scheduled"

    room_id: UUID = Field(...)
    schedule_days: list[str] = Field(...)
    start_time: str = Field(...)
    end_time: str = Field(...)


class StudentEnrolledEvent(DomainEvent):
    """
    Event emitted when a student successfully enrolls in a section.

    This is a critical event for enrollment aggregate.
    """

    EVENT_TYPE: ClassVar[str] = "academic.enrollment.student_enrolled"

    student_id: UUID = Field(...)
    section_id: UUID = Field(...)
    course_code: str = Field(...)
    enrolled_at: datetime = Field(default_factory=datetime.utcnow)
    was_on_waitlist: bool = Field(default=False)


class StudentWaitlistedEvent(DomainEvent):
    """Event emitted when a student is added to waitlist."""

    EVENT_TYPE: ClassVar[str] = "academic.enrollment.student_waitlisted"

    student_id: UUID = Field(...)
    section_id: UUID = Field(...)
    waitlist_position: int = Field(...)
    added_at: datetime = Field(default_factory=datetime.utcnow)


class StudentDroppedEvent(DomainEvent):
    """Event emitted when a student drops a course."""

    EVENT_TYPE: ClassVar[str] = "academic.enrollment.student_dropped"

    student_id: UUID = Field(...)
    section_id: UUID = Field(...)
    dropped_at: datetime = Field(default_factory=datetime.utcnow)
    refund_eligible: bool = Field(default=False)


class GradeAssignedEvent(DomainEvent):
    """
    Event emitted when a grade is assigned to a student.

    Grades are immutable, so this event represents the creation of a grade record.
    """

    EVENT_TYPE: ClassVar[str] = "academic.grade.assigned"

    grade_id: UUID = Field(...)
    student_id: UUID = Field(...)
    assessment_id: UUID = Field(...)
    section_id: UUID = Field(...)
    points_earned: float = Field(...)
    total_points: float = Field(...)
    percentage: float = Field(...)
    letter_grade: str = Field(...)
    graded_by: UUID = Field(...)
    graded_at: datetime = Field(default_factory=datetime.utcnow)


class GradeUpdatedEvent(DomainEvent):
    """
    Event emitted when a grade is regraded.

    Since grades are immutable, this creates a new grade version.
    """

    EVENT_TYPE: ClassVar[str] = "academic.grade.updated"

    old_grade_id: UUID = Field(...)
    new_grade_id: UUID = Field(...)
    student_id: UUID = Field(...)
    old_points: float = Field(...)
    new_points: float = Field(...)
    reason: str = Field(...)


class EnrollmentPolicyViolatedEvent(DomainEvent):
    """Event emitted when an enrollment policy is violated."""

    EVENT_TYPE: ClassVar[str] = "academic.enrollment.policy_violated"

    student_id: UUID = Field(...)
    section_id: UUID = Field(...)
    policy_type: str = Field(...)
    violation_reason: str = Field(...)
    attempted_at: datetime = Field(default_factory=datetime.utcnow)


class PrerequisiteCheckFailedEvent(DomainEvent):
    """Event emitted when prerequisite check fails."""

    EVENT_TYPE: ClassVar[str] = "academic.enrollment.prerequisite_failed"

    student_id: UUID = Field(...)
    course_id: UUID = Field(...)
    missing_prerequisites: list[str] = Field(...)

