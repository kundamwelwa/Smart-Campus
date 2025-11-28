"""
Academic Service Database Models

SQLAlchemy models for courses, sections, enrollments, and grades.
"""

from datetime import date, datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from shared.database.postgres import Base


class CourseModel(Base):
    """Course database model."""

    __tablename__ = "courses"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    course_code: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    credits: Mapped[int] = mapped_column(Integer, nullable=False)
    level: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    department: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    prerequisites: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    corequisites: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    learning_outcomes: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    is_lab_required: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    max_enrollment_default: Mapped[int] = mapped_column(Integer, default=30, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)


class SectionModel(Base):
    """Section database model."""

    __tablename__ = "sections"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    course_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("courses.id"), nullable=False, index=True
    )
    section_number: Mapped[str] = mapped_column(String(10), nullable=False)
    semester: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    instructor_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)

    # Schedule
    schedule_days: Mapped[list] = mapped_column(JSON, nullable=False)
    start_time: Mapped[str] = mapped_column(String(5), nullable=False)  # HH:MM
    end_time: Mapped[str] = mapped_column(String(5), nullable=False)  # HH:MM
    room_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)

    # Enrollment
    max_enrollment: Mapped[int] = mapped_column(Integer, default=30, nullable=False)
    current_enrollment: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    waitlist_size: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_waitlist: Mapped[int] = mapped_column(Integer, default=10, nullable=False)

    # Dates
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    add_drop_deadline: Mapped[date] = mapped_column(Date, nullable=False)
    withdrawal_deadline: Mapped[date] = mapped_column(Date, nullable=False)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)


class EnrollmentModel(Base):
    """Enrollment database model."""

    __tablename__ = "enrollments"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    student_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)
    section_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("sections.id"), nullable=False, index=True
    )

    # Status
    enrollment_status: Mapped[str] = mapped_column(
        String(20), default="enrolled", nullable=False, index=True
    )
    is_waitlisted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    waitlist_position: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Academic Performance
    current_grade_percentage: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    current_letter_grade: Mapped[str | None] = mapped_column(String(2), nullable=True)
    attendance_percentage: Mapped[float] = mapped_column(Float, default=100.0, nullable=False)

    # ML Predictions
    dropout_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_final_grade: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Timestamps
    enrolled_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class GradeModel(Base):
    """
    Grade database model (immutable).

    Grades are never updated, only new versions are created.
    """

    __tablename__ = "grades"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    student_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)
    assessment_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)
    section_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)

    # Grade Data (encrypted in database)
    points_earned: Mapped[str] = mapped_column(Text, nullable=False, comment="Encrypted")
    total_points: Mapped[str] = mapped_column(Text, nullable=False, comment="Encrypted")
    percentage: Mapped[str] = mapped_column(Text, nullable=False, comment="Encrypted")
    letter_grade: Mapped[str] = mapped_column(String(2), nullable=False)

    # Metadata
    graded_by: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    graded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_late: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Feedback (encrypted)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True, comment="Encrypted")

    # Versioning for regrades
    previous_grade_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class AssignmentModel(Base):
    """Assignment for a course section (auto-gradable)."""

    __tablename__ = "assignments"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    course_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("courses.id"), nullable=False, index=True
    )
    section_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("sections.id"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    type: Mapped[str] = mapped_column(String(50), default="auto", nullable=False)
    due_date: Mapped[date] = mapped_column(Date, nullable=False)
    total_points: Mapped[int] = mapped_column(Integer, default=100, nullable=False)

    # External grader linkage (e.g. INGInious task id)
    external_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_by_lecturer_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)

    status: Mapped[str] = mapped_column(
        String(20), default="active", nullable=False, index=True
    )  # active, archived

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class SubmissionModel(Base):
    """Student submission for an assignment."""

    __tablename__ = "submissions"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    assignment_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("assignments.id"), nullable=False, index=True
    )
    student_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)

    submitted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    # Raw answer payload (JSON-serializable structure as text)
    raw_answer: Mapped[str] = mapped_column(Text, nullable=False)

    # Auto-grader output
    auto_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    auto_feedback: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Lecturer approval / override
    lecturer_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    lecturer_feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    approved_by_lecturer_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    status: Mapped[str] = mapped_column(
        String(20), default="submitted", nullable=False, index=True
    )  # submitted, auto_graded, approved

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )


class QuestionModel(Base):
    """Question bank for assignments (rule questions and course content questions)."""

    __tablename__ = "questions"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    course_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("courses.id"), nullable=True, index=True
    )  # None = rule question (applies to all courses)

    question_type: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True
    )  # rule, course_content

    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    question_format: Mapped[str] = mapped_column(
        String(20), default="multiple_choice", nullable=False
    )  # multiple_choice, true_false, short_answer

    # Options for multiple choice / true-false
    options: Mapped[dict] = mapped_column(JSON, nullable=True)  # {"A": "option1", "B": "option2", ...}
    correct_answer: Mapped[str] = mapped_column(Text, nullable=False)  # "A", "B", "true", "false", or text

    points: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    # Metadata
    created_by_lecturer_id: Mapped[UUID | None] = mapped_column(PGUUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)


class AssignmentQuestionModel(Base):
    """Links questions to assignments with random selection settings."""

    __tablename__ = "assignment_questions"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    assignment_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("assignments.id"), nullable=False, index=True
    )
    question_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("questions.id"), nullable=False, index=True
    )

    # Random selection settings
    is_random: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    display_order: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class AnswerModel(Base):
    """Student answer to a specific question within a submission."""

    __tablename__ = "answers"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    submission_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("submissions.id"), nullable=False, index=True
    )
    question_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("questions.id"), nullable=False, index=True
    )
    student_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)

    # Student's answer
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Auto-grading result
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    points_earned: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    points_possible: Mapped[float] = mapped_column(Float, nullable=False)
    auto_feedback: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Lecturer override
    lecturer_points_override: Mapped[float | None] = mapped_column(Float, nullable=True)
    lecturer_feedback: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
