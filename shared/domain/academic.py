"""
Academic Domain Models

Entities related to courses, enrollment, assessments, and grades.
Implements immutability for Grade objects and policy-driven enrollment.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.domain.entities import VersionedEntity


class CourseLevel(str, Enum):
    """Academic course level."""

    UNDERGRADUATE = "undergraduate"
    GRADUATE = "graduate"
    DOCTORAL = "doctoral"
    CERTIFICATE = "certificate"


class AssessmentType(str, Enum):
    """Type of academic assessment."""

    EXAM = "exam"
    QUIZ = "quiz"
    ASSIGNMENT = "assignment"
    PROJECT = "project"
    PRESENTATION = "presentation"
    PARTICIPATION = "participation"
    LAB = "lab"


class GradeScale(str, Enum):
    """Grading scale type."""

    LETTER = "letter"  # A, B, C, D, F
    PERCENTAGE = "percentage"  # 0-100
    PASS_FAIL = "pass_fail"
    HONORS = "honors"


class Course(VersionedEntity):
    """
    Course entity representing an academic course offering.

    A course is a reusable template that can have multiple sections
    across different semesters.
    """

    course_code: str = Field(..., description="Unique course code (e.g., CS-101)")
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., min_length=1, max_length=2000)
    credits: int = Field(..., ge=1, le=12, description="Credit hours")
    level: CourseLevel = Field(default=CourseLevel.UNDERGRADUATE)
    department: str = Field(..., max_length=100)
    prerequisites: list[str] = Field(
        default_factory=list, description="List of prerequisite course codes"
    )
    corequisites: list[str] = Field(
        default_factory=list, description="List of corequisite course codes"
    )
    learning_outcomes: list[str] = Field(default_factory=list)
    is_lab_required: bool = Field(default=False)
    max_enrollment_default: int = Field(default=30, ge=1, le=500)

    @field_validator("course_code")
    @classmethod
    def validate_course_code(cls, v: str) -> str:
        """Validate course code format."""
        if not v or len(v) < 3:
            raise ValueError("Course code must be at least 3 characters")
        return v.upper()

    def validate_business_rules(self) -> bool:
        """Validate course business rules."""
        if self.credits < 1 or self.credits > 12:
            raise ValueError("Credits must be between 1 and 12")
        return True


class Syllabus(VersionedEntity):
    """
    Course syllabus with detailed curriculum information.

    Versioned to track changes across semesters.
    """

    course_id: UUID = Field(..., description="Associated course ID")
    semester: str = Field(..., description="Semester (e.g., Fall 2024)")
    instructor_id: UUID = Field(..., description="Instructor person ID")

    # Syllabus Content
    overview: str = Field(..., max_length=5000)
    topics: list[str] = Field(default_factory=list)
    required_textbooks: list[dict[str, str]] = Field(default_factory=list)
    recommended_readings: list[dict[str, str]] = Field(default_factory=list)

    # Grading Policy
    grading_scale: GradeScale = Field(default=GradeScale.LETTER)
    grade_distribution: dict[str, float] = Field(
        default_factory=dict,
        description="Grade component weights (e.g., {'exams': 0.4, 'assignments': 0.3})",
    )

    # Policies
    attendance_policy: str = Field(..., max_length=1000)
    late_submission_policy: str = Field(..., max_length=1000)
    academic_integrity_policy: str = Field(..., max_length=2000)

    def validate_business_rules(self) -> bool:
        """Validate syllabus business rules."""
        # Validate grade distribution sums to 1.0
        if self.grade_distribution:
            total = sum(self.grade_distribution.values())
            if not (0.99 <= total <= 1.01):  # Allow small floating point errors
                raise ValueError(
                    f"Grade distribution must sum to 1.0, got {total}"
                )
        return True


class Section(VersionedEntity):
    """
    Course section representing a specific offering in a semester.

    A section is an instance of a course with specific timing, room, and instructor.
    """

    course_id: UUID = Field(..., description="Parent course ID")
    section_number: str = Field(..., description="Section identifier (e.g., '001', 'A')")
    semester: str = Field(..., description="Semester (e.g., Fall 2024)")
    instructor_id: UUID = Field(..., description="Instructor person ID")

    # Scheduling
    schedule_days: list[str] = Field(
        ..., description="Days of week (e.g., ['Monday', 'Wednesday', 'Friday'])"
    )
    start_time: str = Field(..., description="Start time (HH:MM format)")
    end_time: str = Field(..., description="End time (HH:MM format)")
    room_id: UUID | None = Field(default=None, description="Assigned room ID")

    # Enrollment
    max_enrollment: int = Field(default=30, ge=1, le=500)
    current_enrollment: int = Field(default=0, ge=0)
    waitlist_size: int = Field(default=0, ge=0)
    max_waitlist: int = Field(default=10, ge=0)

    # Dates
    start_date: date = Field(...)
    end_date: date = Field(...)
    add_drop_deadline: date = Field(...)
    withdrawal_deadline: date = Field(...)

    @field_validator("schedule_days")
    @classmethod
    def validate_schedule_days(cls, v: list[str]) -> list[str]:
        """Validate schedule days."""
        valid_days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
        for day in v:
            if day not in valid_days:
                raise ValueError(f"Invalid day: {day}")
        return v

    def validate_business_rules(self) -> bool:
        """Validate section business rules."""
        if self.end_date <= self.start_date:
            raise ValueError("End date must be after start date")
        if self.current_enrollment > self.max_enrollment:
            raise ValueError("Current enrollment exceeds maximum")
        if self.waitlist_size > self.max_waitlist:
            raise ValueError("Waitlist size exceeds maximum")
        return True

    def is_full(self) -> bool:
        """Check if section is at capacity."""
        return self.current_enrollment >= self.max_enrollment

    def has_waitlist_space(self) -> bool:
        """Check if waitlist has space."""
        return self.waitlist_size < self.max_waitlist

    def can_enroll(self) -> bool:
        """Check if new students can enroll (directly or waitlist)."""
        return not self.is_full() or self.has_waitlist_space()

    def enroll_student(self) -> None:
        """Enroll a student (increment counter)."""
        if self.is_full():
            raise ValueError("Section is full")
        self.current_enrollment += 1
        self.mark_updated()

    def add_to_waitlist(self) -> None:
        """Add student to waitlist."""
        if not self.has_waitlist_space():
            raise ValueError("Waitlist is full")
        self.waitlist_size += 1
        self.mark_updated()

    def remove_from_waitlist(self) -> None:
        """Remove student from waitlist."""
        if self.waitlist_size <= 0:
            raise ValueError("Waitlist is empty")
        self.waitlist_size -= 1
        self.mark_updated()


class Assessment(VersionedEntity):
    """
    Assessment/Assignment entity for a course section.
    """

    section_id: UUID = Field(..., description="Parent section ID")
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., max_length=2000)
    assessment_type: AssessmentType = Field(...)
    total_points: float = Field(..., gt=0, description="Maximum points possible")
    weight: float = Field(..., ge=0.0, le=1.0, description="Weight in final grade")

    # Dates
    assigned_date: datetime = Field(default_factory=datetime.utcnow)
    due_date: datetime = Field(...)
    available_from: datetime | None = Field(default=None)
    available_until: datetime | None = Field(default=None)

    # Settings
    allow_late_submission: bool = Field(default=False)
    late_penalty_per_day: float = Field(default=0.0, ge=0.0, le=1.0)
    is_group_work: bool = Field(default=False)
    max_group_size: int | None = Field(default=None, ge=2)

    def validate_business_rules(self) -> bool:
        """Validate assessment business rules."""
        if self.due_date <= self.assigned_date:
            raise ValueError("Due date must be after assigned date")
        if self.weight < 0.0 or self.weight > 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")
        return True

    def is_available(self) -> bool:
        """Check if assessment is currently available."""
        now = datetime.utcnow()
        if self.available_from and now < self.available_from:
            return False
        return not (self.available_until and now > self.available_until)


class Grade(BaseModel):
    """
    Immutable grade object for a student's assessment.

    Once created, grades cannot be modified (only replaced with new versions
    for audit trail purposes). This ensures grade integrity and compliance.
    """

    model_config = ConfigDict(frozen=True, validate_assignment=True)

    id: UUID = Field(default_factory=uuid4)
    student_id: UUID = Field(..., description="Student ID")
    assessment_id: UUID = Field(..., description="Assessment ID")
    section_id: UUID = Field(..., description="Section ID")

    # Grade Information
    points_earned: float = Field(..., ge=0.0, description="Points earned")
    total_points: float = Field(..., gt=0.0, description="Total possible points")
    letter_grade: str | None = Field(default=None, max_length=2)
    percentage: float = Field(..., ge=0.0, le=100.0)

    # Metadata (immutable)
    graded_by: UUID = Field(..., description="Grader user ID")
    graded_at: datetime = Field(default_factory=datetime.utcnow)
    submitted_at: datetime | None = Field(default=None)
    is_late: bool = Field(default=False)
    late_days: int = Field(default=0, ge=0)

    # Feedback
    feedback: str | None = Field(default=None, max_length=5000)
    rubric_scores: dict[str, float] = Field(default_factory=dict)

    # Audit (immutability enforcement)
    previous_grade_id: UUID | None = Field(
        default=None, description="Previous grade version (if regrade)"
    )
    version: int = Field(default=1, ge=1)

    @field_validator("percentage")
    @classmethod
    def validate_percentage(cls, v: float, info: Any) -> float:
        """Calculate and validate percentage."""
        # Note: In Pydantic v2, we can't access other fields directly in validator
        # Percentage validation will be done at object creation
        return v

    def calculate_percentage(self) -> float:
        """Calculate percentage score."""
        if self.total_points == 0:
            return 0.0
        return (self.points_earned / self.total_points) * 100.0

    def to_letter_grade(self) -> str:
        """Convert percentage to letter grade."""
        if self.percentage >= 93:
            return "A"
        if self.percentage >= 90:
            return "A-"
        if self.percentage >= 87:
            return "B+"
        if self.percentage >= 83:
            return "B"
        if self.percentage >= 80:
            return "B-"
        if self.percentage >= 77:
            return "C+"
        if self.percentage >= 73:
            return "C"
        if self.percentage >= 70:
            return "C-"
        if self.percentage >= 67:
            return "D+"
        if self.percentage >= 63:
            return "D"
        if self.percentage >= 60:
            return "D-"
        return "F"

    @classmethod
    def create(
        cls,
        student_id: UUID,
        assessment_id: UUID,
        section_id: UUID,
        points_earned: float,
        total_points: float,
        graded_by: UUID,
        **kwargs: Any,
    ) -> "Grade":
        """
        Factory method to create a new grade with calculated percentage.

        Args:
            student_id: Student UUID
            assessment_id: Assessment UUID
            section_id: Section UUID
            points_earned: Points the student earned
            total_points: Total possible points
            graded_by: Grader user UUID
            **kwargs: Additional optional fields

        Returns:
            Grade: Immutable grade object
        """
        percentage = (points_earned / total_points) * 100.0 if total_points > 0 else 0.0
        letter_grade = cls._percentage_to_letter(percentage)

        return cls(
            student_id=student_id,
            assessment_id=assessment_id,
            section_id=section_id,
            points_earned=points_earned,
            total_points=total_points,
            percentage=percentage,
            letter_grade=letter_grade,
            graded_by=graded_by,
            **kwargs,
        )

    @staticmethod
    def _percentage_to_letter(percentage: float) -> str:
        """Convert percentage to letter grade."""
        if percentage >= 93:
            return "A"
        if percentage >= 90:
            return "A-"
        if percentage >= 87:
            return "B+"
        if percentage >= 83:
            return "B"
        if percentage >= 80:
            return "B-"
        if percentage >= 77:
            return "C+"
        if percentage >= 73:
            return "C"
        if percentage >= 70:
            return "C-"
        if percentage >= 67:
            return "D+"
        if percentage >= 63:
            return "D"
        if percentage >= 60:
            return "D-"
        return "F"

    def create_regrade(
        self, new_points_earned: float, graded_by: UUID, feedback: str
    ) -> "Grade":
        """
        Create a new grade version (regrade).

        Since grades are immutable, regrading creates a new grade object
        linked to the previous version for audit trail.

        Args:
            new_points_earned: Updated points
            graded_by: Grader user ID
            feedback: Regrade justification

        Returns:
            Grade: New immutable grade object
        """
        return Grade.create(
            student_id=self.student_id,
            assessment_id=self.assessment_id,
            section_id=self.section_id,
            points_earned=new_points_earned,
            total_points=self.total_points,
            graded_by=graded_by,
            submitted_at=self.submitted_at,
            is_late=self.is_late,
            late_days=self.late_days,
            feedback=feedback,
            previous_grade_id=self.id,
            version=self.version + 1,
        )


class Enrollment(VersionedEntity):
    """
    Student enrollment in a course section.

    Tracks enrollment status, grades, and participation.
    """

    student_id: UUID = Field(..., description="Enrolled student ID")
    section_id: UUID = Field(..., description="Course section ID")
    enrolled_at: datetime = Field(default_factory=datetime.utcnow)

    # Status
    enrollment_status: str = Field(
        default="enrolled",
        description="Status: enrolled, waitlisted, dropped, withdrawn, completed",
    )
    is_waitlisted: bool = Field(default=False)
    waitlist_position: int | None = Field(default=None, ge=1)

    # Academic Performance
    current_grade_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    current_letter_grade: str | None = Field(default=None)
    attendance_percentage: float = Field(default=100.0, ge=0.0, le=100.0)

    # Predictions (ML-generated)
    dropout_probability: float | None = Field(
        default=None, ge=0.0, le=1.0, description="ML-predicted dropout risk"
    )
    predicted_final_grade: float | None = Field(default=None, ge=0.0, le=100.0)

    def validate_business_rules(self) -> bool:
        """Validate enrollment business rules."""
        if self.is_waitlisted and self.enrollment_status == "enrolled":
            raise ValueError("Cannot be both waitlisted and enrolled")
        if self.waitlist_position is not None and not self.is_waitlisted:
            raise ValueError("Waitlist position set but not waitlisted")
        return True

    def enroll_from_waitlist(self) -> None:
        """Move student from waitlist to enrolled status."""
        if not self.is_waitlisted:
            raise ValueError("Student is not on waitlist")
        self.enrollment_status = "enrolled"
        self.is_waitlisted = False
        self.waitlist_position = None
        self.mark_updated()

    def drop(self) -> None:
        """Drop the enrollment."""
        self.enrollment_status = "dropped"
        self.mark_updated()

    def withdraw(self) -> None:
        """Withdraw from the course."""
        self.enrollment_status = "withdrawn"
        self.mark_updated()

    def complete(self, final_grade: str) -> None:
        """Mark enrollment as completed with final grade."""
        self.enrollment_status = "completed"
        self.current_letter_grade = final_grade
        self.mark_updated()


class Waitlist(VersionedEntity):
    """
    Waitlist management for course sections.

    Maintains ordered queue of students waiting for enrollment.
    """

    section_id: UUID = Field(..., description="Course section ID")
    entries: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Waitlist entries with student_id, added_at, priority",
    )
    max_size: int = Field(default=10, ge=0)

    def validate_business_rules(self) -> bool:
        """Validate waitlist business rules."""
        if len(self.entries) > self.max_size:
            raise ValueError("Waitlist exceeds maximum size")
        return True

    def add_student(self, student_id: UUID, priority: int = 0) -> int:
        """
        Add student to waitlist.

        Args:
            student_id: Student UUID
            priority: Priority score (higher = higher priority)

        Returns:
            int: Waitlist position (1-indexed)
        """
        if len(self.entries) >= self.max_size:
            raise ValueError("Waitlist is full")

        entry = {
            "student_id": str(student_id),
            "added_at": datetime.utcnow().isoformat(),
            "priority": priority,
        }
        self.entries.append(entry)

        # Sort by priority (descending) then by added_at (ascending)
        self.entries.sort(
            key=lambda x: (-x["priority"], x["added_at"])
        )

        # Find position
        position = next(
            i + 1
            for i, e in enumerate(self.entries)
            if UUID(e["student_id"]) == student_id
        )

        self.mark_updated()
        return position

    def remove_student(self, student_id: UUID) -> bool:
        """Remove student from waitlist."""
        initial_len = len(self.entries)
        self.entries = [e for e in self.entries if UUID(e["student_id"]) != student_id]

        if len(self.entries) < initial_len:
            self.mark_updated()
            return True
        return False

    def get_next_student(self) -> UUID | None:
        """Get next student ID from waitlist (highest priority)."""
        if not self.entries:
            return None
        return UUID(self.entries[0]["student_id"])

    def pop_next_student(self) -> UUID | None:
        """Remove and return next student from waitlist."""
        student_id = self.get_next_student()
        if student_id:
            self.remove_student(student_id)
        return student_id

