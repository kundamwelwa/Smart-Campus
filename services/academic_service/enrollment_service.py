"""
Enrollment Service

Core enrollment business logic with policy engine and event sourcing.
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.academic_service.aggregates import EnrollmentAggregate
from services.academic_service.models import CourseModel, EnrollmentModel, SectionModel
from services.user_service.models import StudentModel
from shared.domain.exceptions import EnrollmentPolicyViolationError
from shared.domain.policies import PolicyEngine
from shared.events.base import Snapshot
from shared.events.store import EventStore
from shared.verification.enrollment_invariants import (
    Section as VerificationSection,
)
from shared.verification.enrollment_invariants import (
    TimeSlot,
    assert_enrollment_invariant,
)

logger = structlog.get_logger(__name__)


class EnrollmentService:
    """
    Service orchestrating student enrollment with policy enforcement.

    Implements:
    - Policy-driven enrollment validation
    - Event-sourced enrollment aggregate
    - Optimistic concurrency control
    - Automatic waitlist management
    """

    def __init__(
        self, db_session: AsyncSession, event_store: EventStore, policy_engine: PolicyEngine
    ):
        """
        Initialize enrollment service.

        Args:
            db_session: Database session
            event_store: Event store for event sourcing
            policy_engine: Policy engine for enrollment validation
        """
        self.db = db_session
        self.event_store = event_store
        self.policy_engine = policy_engine

    async def enroll_student(
        self, student_id: UUID, section_id: UUID, user_id: UUID
    ) -> EnrollmentModel:
        """
        Enroll a student in a section with policy validation.

        Process:
        1. Fetch student and section data
        2. Build policy evaluation context
        3. Execute all enrollment policies
        4. If allowed, create enrollment or add to waitlist
        5. Emit domain events
        6. Update read model (database)

        Args:
            student_id: Student UUID
            section_id: Section UUID
            user_id: User performing enrollment (for audit)

        Returns:
            EnrollmentModel: Created enrollment

        Raises:
            EnrollmentPolicyViolationError: If policies reject enrollment
            ValueError: If student or section not found
        """
        logger.info(
            "Starting enrollment process",
            student_id=str(student_id),
            section_id=str(section_id),
        )

        # Fetch section with course data
        section = await self._get_section(section_id)
        if section is None:
            raise ValueError(f"Section not found: {section_id}")

        course = await self._get_course(section.course_id)
        if course is None:
            raise ValueError(f"Course not found: {section.course_id}")

        # Fetch student data
        student = await self._get_student(student_id)
        if student is None:
            raise ValueError(f"Student not found: {student_id}")

        # Check for existing enrollment
        existing = await self._check_existing_enrollment(student_id, section_id)
        if existing:
            raise ValueError("Student already enrolled in this section")

        # Build policy evaluation context
        context = await self._build_policy_context(student_id, section_id, student, section, course)

        # Evaluate all policies
        allowed, policy_results = await self.policy_engine.evaluate_all(
            student_id, section_id, context
        )

        if not allowed:
            # Find first failed policy
            failed_result = next((r for r in policy_results if not r.allowed), None)
            logger.warning(
                "Enrollment denied by policy",
                student_id=str(student_id),
                section_id=str(section_id),
                reason=failed_result.reason if failed_result else "Unknown",
            )
            raise EnrollmentPolicyViolationError(
                reason=failed_result.reason if failed_result else "Policy violation",
                violated_rules=failed_result.violated_rules if failed_result else [],
            )

        # Policies passed - proceed with enrollment
        logger.info(
            "Policies passed, proceeding with enrollment",
            student_id=str(student_id),
            section_id=str(section_id),
        )

        # FORMAL VERIFICATION: Assert enrollment invariant
        # Critical invariant: "No student can be enrolled in overlapping sections
        # that are scheduled at the same time with the same seat allocation."
        try:
            # Build verification sections from database
            verification_sections = await self._build_verification_sections(
                student_id, section_id, section
            )

            # Assert invariant (raises AssertionError if violated)
            assert_enrollment_invariant(
                student_id=int(student_id),
                section_id=int(section_id),
                sections=verification_sections,
                raise_on_violation=True,
            )

            logger.info(
                "Formal verification passed - enrollment invariant satisfied",
                student_id=str(student_id),
                section_id=str(section_id),
            )
        except AssertionError as e:
            logger.error(
                "Formal verification failed - enrollment invariant violated",
                student_id=str(student_id),
                section_id=str(section_id),
                error=str(e),
            )
            raise EnrollmentPolicyViolationError(
                reason=f"Formal verification failed: {str(e)}",
                violated_rules=["enrollment_invariant_time_overlap"],
            )

        # Create enrollment aggregate
        enrollment_id = uuid4()
        aggregate = EnrollmentAggregate(enrollment_id)

        # Determine if direct enrollment or waitlist
        if section.current_enrollment < section.max_enrollment:
            # Direct enrollment
            aggregate.enroll_student(
                student_id=student_id,
                section_id=section_id,
                course_code=course.course_code,
                user_id=user_id,
            )

            # Update section capacity
            section.current_enrollment += 1

        else:
            # Add to waitlist
            if section.waitlist_size >= section.max_waitlist:
                raise ValueError("Section and waitlist are both full")

            waitlist_position = section.waitlist_size + 1
            aggregate.add_to_waitlist(
                student_id=student_id,
                section_id=section_id,
                position=waitlist_position,
                user_id=user_id,
            )

            # Update waitlist size
            section.waitlist_size += 1

        # Persist events to event store
        stream_id = f"enrollment-{enrollment_id}"
        for event in aggregate.get_uncommitted_events():
            await self.event_store.append(
                event=event, stream_id=stream_id, expected_version=None
            )

        aggregate.mark_events_committed()

        # Create snapshot (optional, for performance)
        snapshot = Snapshot(
            aggregate_id=enrollment_id,
            aggregate_type=aggregate.aggregate_type(),
            state=aggregate.get_state(),
            version=aggregate.version,
            event_count=aggregate.version,
        )
        await self.event_store.save_snapshot(snapshot)

        # Update read model (database) for query performance
        enrollment = EnrollmentModel(
            id=enrollment_id,
            student_id=student_id,
            section_id=section_id,
            enrollment_status="waitlisted" if aggregate.is_waitlisted else "enrolled",
            is_waitlisted=aggregate.is_waitlisted,
            waitlist_position=aggregate.waitlist_position,
            enrolled_at=aggregate.enrolled_at or datetime.utcnow(),
        )

        self.db.add(enrollment)
        await self.db.flush()

        logger.info(
            "Enrollment completed",
            enrollment_id=str(enrollment_id),
            student_id=str(student_id),
            section_id=str(section_id),
            status=enrollment.enrollment_status,
        )

        return enrollment

    async def _get_section(self, section_id: UUID) -> SectionModel | None:
        """Fetch section from database."""
        result = await self.db.execute(select(SectionModel).where(SectionModel.id == section_id))
        return result.scalar_one_or_none()

    async def _get_course(self, course_id: UUID) -> CourseModel | None:
        """Fetch course from database."""
        result = await self.db.execute(select(CourseModel).where(CourseModel.id == course_id))
        return result.scalar_one_or_none()

    async def _get_student(self, student_id: UUID) -> StudentModel | None:
        """Fetch student from database."""
        # Note: This requires importing from user_service
        # In microservices, this would be an RPC call
        from services.user_service.models import StudentModel as UserStudentModel

        result = await self.db.execute(
            select(UserStudentModel).where(UserStudentModel.user_id == student_id)
        )
        return result.scalar_one_or_none()

    async def _check_existing_enrollment(
        self, student_id: UUID, section_id: UUID
    ) -> EnrollmentModel | None:
        """Check if enrollment already exists."""
        result = await self.db.execute(
            select(EnrollmentModel).where(
                EnrollmentModel.student_id == student_id,
                EnrollmentModel.section_id == section_id,
                EnrollmentModel.enrollment_status.in_(["enrolled", "waitlisted"]),
            )
        )
        return result.scalar_one_or_none()

    async def _build_policy_context(
        self,
        student_id: UUID,
        section_id: UUID,
        student: Any,
        section: SectionModel,
        course: CourseModel,
    ) -> dict[str, Any]:
        """
        Build context for policy evaluation.

        Gathers all necessary data for policy decisions.
        """
        # Get student's completed courses
        completed_courses = await self._get_completed_courses(student_id)

        # Get student's current enrollments for schedule conflict check
        current_enrollments = await self._get_current_enrollments(student_id, section.semester)

        # Build schedule for current section
        section_schedule = {
            "days": section.schedule_days,
            "start_time": section.start_time,
            "end_time": section.end_time,
        }

        # Build context
        return {
            # Course data
            "course_prerequisites": course.prerequisites,
            "course_credits": course.credits,
            "course_code": course.course_code,
            # Section data
            "section_max_enrollment": section.max_enrollment,
            "section_current_enrollment": section.current_enrollment,
            "section_schedule": section_schedule,
            # Student data
            "student_completed_courses": completed_courses,
            "student_current_credits": await self._calculate_current_credits(
                student_id, section.semester
            ),
            "student_gpa": student.gpa if student else 0.0,
            "student_academic_standing": student.academic_standing if student else "good",
            "student_current_schedule": current_enrollments,
            # Temporal data
            "current_time": datetime.utcnow(),
        }


    async def _get_completed_courses(self, student_id: UUID) -> list[str]:
        """Get list of course codes the student has completed."""
        result = await self.db.execute(
            select(CourseModel.course_code)
            .join(SectionModel, SectionModel.course_id == CourseModel.id)
            .join(
                EnrollmentModel,
                EnrollmentModel.section_id == SectionModel.id,
            )
            .where(
                EnrollmentModel.student_id == student_id,
                EnrollmentModel.enrollment_status == "completed",
            )
        )
        return [row[0] for row in result.all()]

    async def _get_current_enrollments(
        self, student_id: UUID, semester: str
    ) -> list[dict[str, Any]]:
        """Get student's current enrollments for schedule conflict checking."""
        result = await self.db.execute(
            select(SectionModel, CourseModel)
            .join(CourseModel, SectionModel.course_id == CourseModel.id)
            .join(EnrollmentModel, EnrollmentModel.section_id == SectionModel.id)
            .where(
                EnrollmentModel.student_id == student_id,
                SectionModel.semester == semester,
                EnrollmentModel.enrollment_status == "enrolled",
            )
        )

        enrollments = []
        for section, course in result.all():
            enrollments.append({
                "section_id": str(section.id),
                "course_code": course.course_code,
                "days": section.schedule_days,
                "start_time": section.start_time,
                "end_time": section.end_time,
            })

        return enrollments

    async def _calculate_current_credits(self, student_id: UUID, semester: str) -> int:
        """Calculate total credits student is currently enrolled in."""
        result = await self.db.execute(
            select(CourseModel.credits)
            .join(SectionModel, SectionModel.course_id == CourseModel.id)
            .join(EnrollmentModel, EnrollmentModel.section_id == SectionModel.id)
            .where(
                EnrollmentModel.student_id == student_id,
                SectionModel.semester == semester,
                EnrollmentModel.enrollment_status == "enrolled",
            )
        )

        return sum(row[0] for row in result.all())

    async def _build_verification_sections(
        self,
        student_id: UUID,
        target_section_id: UUID,
        target_section: SectionModel,
    ) -> dict[int, VerificationSection]:
        """
        Build verification Section objects from database models.

        This converts database models to the verification format needed
        for formal invariant checking.

        Args:
            student_id: Student being enrolled
            target_section_id: Section being enrolled in
            target_section: SectionModel for target section

        Returns:
            Dictionary of section_id -> VerificationSection
        """
        from datetime import time as dt_time

        sections = {}

        # Get all sections the student is currently enrolled in
        current_enrollments = await self._get_current_enrollments(
            student_id, target_section.semester
        )

        # Convert each enrolled section
        for enrollment in current_enrollments:
            section_id = int(UUID(enrollment['section_id']))
            section_model = await self._get_section(UUID(enrollment['section_id']))

            if section_model:
                # Parse schedule days
                days = set(enrollment.get('days', '').split(',')) if enrollment.get('days') else set()

                # Parse times
                start_time = enrollment.get('start_time')
                end_time = enrollment.get('end_time')

                if isinstance(start_time, str):
                    start_time = dt_time.fromisoformat(start_time)
                if isinstance(end_time, str):
                    end_time = dt_time.fromisoformat(end_time)

                time_slot = TimeSlot(
                    start_time=start_time or dt_time(9, 0),
                    end_time=end_time or dt_time(10, 0),
                    days=days,
                )

                # Get enrolled students for this section
                enrollments_result = await self.db.execute(
                    select(EnrollmentModel.student_id).where(
                        EnrollmentModel.section_id == section_model.id,
                        EnrollmentModel.enrollment_status == "enrolled",
                    )
                )
                enrolled_students = {int(uid) for uid, in enrollments_result.all()}

                sections[section_id] = VerificationSection(
                    section_id=section_id,
                    course_id=int(section_model.course_id),
                    room_id=int(section_model.room_id) if section_model.room_id else 0,
                    capacity=section_model.max_enrollment,
                    time_slot=time_slot,
                    enrolled_students=enrolled_students,
                )

        # Add target section
        target_section_id_int = int(target_section_id)
        days = set(target_section.schedule_days.split(',')) if target_section.schedule_days else set()

        start_time = target_section.start_time
        end_time = target_section.end_time

        if isinstance(start_time, str):
            start_time = dt_time.fromisoformat(start_time)
        if isinstance(end_time, str):
            end_time = dt_time.fromisoformat(end_time)

        time_slot = TimeSlot(
            start_time=start_time or dt_time(9, 0),
            end_time=end_time or dt_time(10, 0),
            days=days,
        )

        # Get enrolled students for target section
        enrollments_result = await self.db.execute(
            select(EnrollmentModel.student_id).where(
                EnrollmentModel.section_id == target_section_id,
                EnrollmentModel.enrollment_status == "enrolled",
            )
        )
        enrolled_students = {int(uid) for uid, in enrollments_result.all()}

        sections[target_section_id_int] = VerificationSection(
            section_id=target_section_id_int,
            course_id=int(target_section.course_id),
            room_id=int(target_section.room_id) if target_section.room_id else 0,
            capacity=target_section.max_enrollment,
            time_slot=time_slot,
            enrolled_students=enrolled_students,
        )

        return sections



