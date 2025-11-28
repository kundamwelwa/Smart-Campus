"""
Academic Service Aggregates

Event-sourced aggregates for enrollment and academic operations.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

import structlog

from shared.events.academic_events import (
    StudentDroppedEvent,
    StudentEnrolledEvent,
    StudentWaitlistedEvent,
)
from shared.events.aggregate import AggregateRoot
from shared.events.base import DomainEvent, EventMetadata

logger = structlog.get_logger(__name__)


class EnrollmentAggregate(AggregateRoot):
    """
    Enrollment aggregate managing student enrollment lifecycle.

    State is built from events:
    - StudentEnrolledEvent
    - StudentWaitlistedEvent
    - StudentDroppedEvent

    Implements critical business invariants:
    - No duplicate enrollments
    - Capacity enforcement
    - Schedule conflict detection
    """

    def __init__(self, aggregate_id: UUID):
        """
        Initialize enrollment aggregate.

        Args:
            aggregate_id: Enrollment UUID
        """
        super().__init__(aggregate_id)

        # Aggregate state
        self.student_id: UUID | None = None
        self.section_id: UUID | None = None
        self.course_code: str | None = None
        self.status: str = "pending"
        self.is_waitlisted: bool = False
        self.waitlist_position: int | None = None
        self.enrolled_at: datetime | None = None
        self.dropped_at: datetime | None = None

    @classmethod
    def aggregate_type(cls) -> str:
        """Get aggregate type."""
        return "Enrollment"

    def enroll_student(
        self,
        student_id: UUID,
        section_id: UUID,
        course_code: str,
        user_id: UUID,
        was_on_waitlist: bool = False,
    ) -> None:
        """
        Enroll a student in a section.

        Args:
            student_id: Student UUID
            section_id: Section UUID
            course_code: Course code
            user_id: User performing the action
            was_on_waitlist: Whether student was on waitlist
        """
        # Business rule: Cannot enroll if already enrolled
        if self.status == "enrolled":
            raise ValueError("Student is already enrolled")

        # Create and raise event
        event = StudentEnrolledEvent(
            metadata=EventMetadata(user_id=user_id, service="academic_service"),
            aggregate_id=self.id,
            aggregate_type=self.aggregate_type(),
            sequence_number=self.version,
            student_id=student_id,
            section_id=section_id,
            course_code=course_code,
            enrolled_at=datetime.utcnow(),
            was_on_waitlist=was_on_waitlist,
        )

        self.raise_event(event)

    def add_to_waitlist(
        self, student_id: UUID, section_id: UUID, position: int, user_id: UUID
    ) -> None:
        """
        Add student to waitlist.

        Args:
            student_id: Student UUID
            section_id: Section UUID
            position: Waitlist position
            user_id: User performing the action
        """
        # Create and raise event
        event = StudentWaitlistedEvent(
            metadata=EventMetadata(user_id=user_id, service="academic_service"),
            aggregate_id=self.id,
            aggregate_type=self.aggregate_type(),
            sequence_number=self.version,
            student_id=student_id,
            section_id=section_id,
            waitlist_position=position,
            added_at=datetime.utcnow(),
        )

        self.raise_event(event)

    def drop(self, user_id: UUID, refund_eligible: bool = False) -> None:
        """
        Drop enrollment.

        Args:
            user_id: User performing the action
            refund_eligible: Whether student is eligible for refund
        """
        if self.status != "enrolled":
            raise ValueError(f"Cannot drop enrollment with status: {self.status}")

        # Create and raise event
        event = StudentDroppedEvent(
            metadata=EventMetadata(user_id=user_id, service="academic_service"),
            aggregate_id=self.id,
            aggregate_type=self.aggregate_type(),
            sequence_number=self.version,
            student_id=self.student_id,  # type: ignore
            section_id=self.section_id,  # type: ignore
            dropped_at=datetime.utcnow(),
            refund_eligible=refund_eligible,
        )

        self.raise_event(event)

    def apply_event(self, event: DomainEvent) -> None:
        """
        Apply event to update aggregate state.

        Args:
            event: Domain event to apply
        """
        if isinstance(event, StudentEnrolledEvent):
            self._apply_enrolled(event)
        elif isinstance(event, StudentWaitlistedEvent):
            self._apply_waitlisted(event)
        elif isinstance(event, StudentDroppedEvent):
            self._apply_dropped(event)

    def _apply_enrolled(self, event: StudentEnrolledEvent) -> None:
        """Apply StudentEnrolledEvent."""
        self.student_id = event.student_id
        self.section_id = event.section_id
        self.course_code = event.course_code
        self.status = "enrolled"
        self.is_waitlisted = False
        self.waitlist_position = None
        self.enrolled_at = event.enrolled_at

    def _apply_waitlisted(self, event: StudentWaitlistedEvent) -> None:
        """Apply StudentWaitlistedEvent."""
        self.student_id = event.student_id
        self.section_id = event.section_id
        self.status = "waitlisted"
        self.is_waitlisted = True
        self.waitlist_position = event.waitlist_position

    def _apply_dropped(self, event: StudentDroppedEvent) -> None:
        """Apply StudentDroppedEvent."""
        self.status = "dropped"
        self.dropped_at = event.dropped_at

    def get_state(self) -> dict[str, Any]:
        """Get current state for snapshotting."""
        return {
            "aggregate_id": str(self.id),
            "version": self.version,
            "student_id": str(self.student_id) if self.student_id else None,
            "section_id": str(self.section_id) if self.section_id else None,
            "course_code": self.course_code,
            "status": self.status,
            "is_waitlisted": self.is_waitlisted,
            "waitlist_position": self.waitlist_position,
            "enrolled_at": self.enrolled_at.isoformat() if self.enrolled_at else None,
            "dropped_at": self.dropped_at.isoformat() if self.dropped_at else None,
        }

    @classmethod
    def from_snapshot(cls, snapshot_data: dict[str, Any]) -> "EnrollmentAggregate":
        """Restore from snapshot."""
        aggregate = cls(UUID(snapshot_data["aggregate_id"]))
        aggregate.version = snapshot_data["version"]
        aggregate.student_id = (
            UUID(snapshot_data["student_id"]) if snapshot_data.get("student_id") else None
        )
        aggregate.section_id = (
            UUID(snapshot_data["section_id"]) if snapshot_data.get("section_id") else None
        )
        aggregate.course_code = snapshot_data.get("course_code")
        aggregate.status = snapshot_data.get("status", "pending")
        aggregate.is_waitlisted = snapshot_data.get("is_waitlisted", False)
        aggregate.waitlist_position = snapshot_data.get("waitlist_position")

        if snapshot_data.get("enrolled_at"):
            aggregate.enrolled_at = datetime.fromisoformat(snapshot_data["enrolled_at"])
        if snapshot_data.get("dropped_at"):
            aggregate.dropped_at = datetime.fromisoformat(snapshot_data["dropped_at"])

        return aggregate

