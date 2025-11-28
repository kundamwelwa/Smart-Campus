"""
Event Sourcing Implementation

Implements event-driven architecture with event store, snapshotting, and replay
for critical subsystems like enrollment. Provides complete audit trail and
time-travel capabilities.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import structlog
from pydantic import BaseModel, ConfigDict, Field

logger = structlog.get_logger(__name__)


class EventType(str, Enum):
    """Types of domain events."""

    # User events
    USER_REGISTERED = "user_registered"
    USER_UPDATED = "user_updated"
    USER_DELETED = "user_deleted"

    # Enrollment events
    ENROLLMENT_CREATED = "enrollment_created"
    ENROLLMENT_CONFIRMED = "enrollment_confirmed"
    ENROLLMENT_DROPPED = "enrollment_dropped"
    WAITLIST_JOINED = "waitlist_joined"
    WAITLIST_PROMOTED = "waitlist_promoted"

    # Grade events
    GRADE_ASSIGNED = "grade_assigned"
    GRADE_UPDATED = "grade_updated"

    # Course events
    COURSE_CREATED = "course_created"
    COURSE_UPDATED = "course_updated"
    SECTION_CREATED = "section_created"
    SECTION_UPDATED = "section_updated"

    # Facility events
    ROOM_BOOKED = "room_booked"
    BOOKING_CANCELLED = "booking_cancelled"

    # Security events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"


class DomainEvent(BaseModel, ABC):
    """
    Abstract base class for all domain events.

    Events are immutable records of things that have happened in the system.
    They form the source of truth for event sourcing.
    """

    model_config = ConfigDict(frozen=True, validate_assignment=True)

    # Event metadata
    event_id: UUID = Field(default_factory=uuid4, description="Unique event ID")
    event_type: EventType = Field(..., description="Type of event")
    aggregate_id: UUID = Field(..., description="ID of aggregate this event affects")
    aggregate_type: str = Field(..., description="Type of aggregate (user, enrollment, etc.)")
    occurred_at: datetime = Field(
        default_factory=datetime.utcnow, description="When event occurred"
    )

    # Actor information
    actor_id: UUID | None = Field(default=None, description="Who caused this event")
    actor_type: str = Field(default="user", description="Type of actor")

    # Causality
    correlation_id: str | None = Field(
        default=None, description="Correlation ID for tracking related events"
    )
    causation_id: UUID | None = Field(
        default=None, description="ID of event that caused this event"
    )

    # Event versioning
    version: int = Field(..., description="Aggregate version after this event", ge=1)

    @abstractmethod
    def get_payload(self) -> dict[str, Any]:
        """
        Get event-specific payload data.

        Returns:
            Dictionary with event data
        """

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for storage."""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type.value,
            "aggregate_id": str(self.aggregate_id),
            "aggregate_type": self.aggregate_type,
            "occurred_at": self.occurred_at.isoformat(),
            "actor_id": str(self.actor_id) if self.actor_id else None,
            "actor_type": self.actor_type,
            "correlation_id": self.correlation_id,
            "causation_id": str(self.causation_id) if self.causation_id else None,
            "version": self.version,
            "payload": self.get_payload(),
        }


class EnrollmentCreatedEvent(DomainEvent):
    """Event emitted when a student enrolls in a section."""

    event_type: EventType = Field(default=EventType.ENROLLMENT_CREATED)
    aggregate_type: str = Field(default="enrollment")

    # Event-specific data
    student_id: UUID = Field(...)
    section_id: UUID = Field(...)
    course_id: UUID = Field(...)
    semester: str = Field(...)
    enrolled_at: datetime = Field(default_factory=datetime.utcnow)
    is_waitlisted: bool = Field(default=False)

    def get_payload(self) -> dict[str, Any]:
        """Get event payload."""
        return {
            "student_id": str(self.student_id),
            "section_id": str(self.section_id),
            "course_id": str(self.course_id),
            "semester": self.semester,
            "enrolled_at": self.enrolled_at.isoformat(),
            "is_waitlisted": self.is_waitlisted,
        }


class EnrollmentDroppedEvent(DomainEvent):
    """Event emitted when a student drops an enrollment."""

    event_type: EventType = Field(default=EventType.ENROLLMENT_DROPPED)
    aggregate_type: str = Field(default="enrollment")

    # Event-specific data
    student_id: UUID = Field(...)
    section_id: UUID = Field(...)
    dropped_at: datetime = Field(default_factory=datetime.utcnow)
    reason: str | None = Field(default=None)

    def get_payload(self) -> dict[str, Any]:
        """Get event payload."""
        return {
            "student_id": str(self.student_id),
            "section_id": str(self.section_id),
            "dropped_at": self.dropped_at.isoformat(),
            "reason": self.reason,
        }


class GradeAssignedEvent(DomainEvent):
    """Event emitted when a grade is assigned."""

    event_type: EventType = Field(default=EventType.GRADE_ASSIGNED)
    aggregate_type: str = Field(default="grade")

    # Event-specific data
    student_id: UUID = Field(...)
    assessment_id: UUID = Field(...)
    section_id: UUID = Field(...)
    points_earned: float = Field(...)
    total_points: float = Field(...)
    percentage: float = Field(...)
    letter_grade: str = Field(...)
    graded_by: UUID = Field(...)

    def get_payload(self) -> dict[str, Any]:
        """Get event payload."""
        return {
            "student_id": str(self.student_id),
            "assessment_id": str(self.assessment_id),
            "section_id": str(self.section_id),
            "points_earned": self.points_earned,
            "total_points": self.total_points,
            "percentage": self.percentage,
            "letter_grade": self.letter_grade,
            "graded_by": str(self.graded_by),
        }


class EventStore:
    """
    Event store for persisting and retrieving domain events.

    Provides append-only storage with querying capabilities.
    """

    def __init__(self):
        """Initialize event store."""
        self.events: list[DomainEvent] = []
        self.events_by_aggregate: dict[UUID, list[DomainEvent]] = {}
        self.events_by_type: dict[EventType, list[DomainEvent]] = {}

    async def append(self, event: DomainEvent) -> None:
        """
        Append an event to the store.

        Args:
            event: Domain event to append
        """
        self.events.append(event)

        # Index by aggregate
        if event.aggregate_id not in self.events_by_aggregate:
            self.events_by_aggregate[event.aggregate_id] = []
        self.events_by_aggregate[event.aggregate_id].append(event)

        # Index by type
        if event.event_type not in self.events_by_type:
            self.events_by_type[event.event_type] = []
        self.events_by_type[event.event_type].append(event)

        logger.info(
            "Event appended",
            event_type=event.event_type.value,
            aggregate_id=str(event.aggregate_id),
            version=event.version,
        )

    async def get_events_for_aggregate(
        self, aggregate_id: UUID, from_version: int = 0
    ) -> list[DomainEvent]:
        """
        Get all events for a specific aggregate.

        Args:
            aggregate_id: Aggregate ID
            from_version: Starting version (0 = all events)

        Returns:
            List of events for the aggregate
        """
        events = self.events_by_aggregate.get(aggregate_id, [])
        if from_version > 0:
            events = [e for e in events if e.version >= from_version]
        return sorted(events, key=lambda e: e.version)

    async def get_events_by_type(self, event_type: EventType) -> list[DomainEvent]:
        """Get all events of a specific type."""
        return self.events_by_type.get(event_type, [])

    async def get_events_in_timerange(
        self, start: datetime, end: datetime
    ) -> list[DomainEvent]:
        """Get all events within a time range."""
        return [e for e in self.events if start <= e.occurred_at <= end]

    async def get_all_events(self) -> list[DomainEvent]:
        """Get all events in chronological order."""
        return sorted(self.events, key=lambda e: e.occurred_at)

    def get_current_version(self, aggregate_id: UUID) -> int:
        """
        Get current version of an aggregate.

        Args:
            aggregate_id: Aggregate ID

        Returns:
            int: Current version (0 if no events)
        """
        events = self.events_by_aggregate.get(aggregate_id, [])
        return max((e.version for e in events), default=0)


class Snapshot(BaseModel):
    """
    Snapshot of aggregate state at a specific version.

    Snapshots optimize event replay by storing periodic state checkpoints.
    """

    aggregate_id: UUID = Field(...)
    aggregate_type: str = Field(...)
    version: int = Field(..., description="Version of aggregate at snapshot time")
    state: dict[str, Any] = Field(..., description="Aggregate state")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SnapshotStore:
    """
    Storage for aggregate snapshots.

    Reduces replay time by storing periodic state checkpoints.
    """

    def __init__(self, snapshot_interval: int = 10):
        """
        Initialize snapshot store.

        Args:
            snapshot_interval: Create snapshot every N events
        """
        self.snapshot_interval = snapshot_interval
        self.snapshots: dict[UUID, Snapshot] = {}

    async def save_snapshot(self, snapshot: Snapshot) -> None:
        """
        Save a snapshot.

        Args:
            snapshot: Snapshot to save
        """
        self.snapshots[snapshot.aggregate_id] = snapshot
        logger.info(
            "Snapshot saved",
            aggregate_id=str(snapshot.aggregate_id),
            version=snapshot.version,
        )

    async def get_snapshot(self, aggregate_id: UUID) -> Snapshot | None:
        """
        Get latest snapshot for an aggregate.

        Args:
            aggregate_id: Aggregate ID

        Returns:
            Latest snapshot or None
        """
        return self.snapshots.get(aggregate_id)

    def should_snapshot(self, current_version: int) -> bool:
        """
        Check if snapshot should be created.

        Args:
            current_version: Current aggregate version

        Returns:
            bool: True if snapshot should be created
        """
        return current_version % self.snapshot_interval == 0


class AggregateRoot(ABC):
    """
    Abstract base class for event-sourced aggregates.

    Aggregates are entities whose state is rebuilt from events.
    """

    def __init__(self, aggregate_id: UUID, aggregate_type: str):
        """
        Initialize aggregate root.

        Args:
            aggregate_id: Unique aggregate ID
            aggregate_type: Type identifier
        """
        self.aggregate_id = aggregate_id
        self.aggregate_type = aggregate_type
        self.version = 0
        self.uncommitted_events: list[DomainEvent] = []

    @abstractmethod
    async def apply_event(self, event: DomainEvent) -> None:
        """
        Apply an event to update aggregate state.

        Args:
            event: Domain event to apply
        """

    async def load_from_history(self, events: list[DomainEvent]) -> None:
        """
        Rebuild aggregate state from event history.

        Args:
            events: Historical events for this aggregate
        """
        for event in sorted(events, key=lambda e: e.version):
            await self.apply_event(event)
            self.version = event.version

        logger.info(
            "Aggregate loaded from history",
            aggregate_id=str(self.aggregate_id),
            version=self.version,
            event_count=len(events),
        )

    def record_event(self, event: DomainEvent) -> None:
        """
        Record a new event (not yet committed).

        Args:
            event: Domain event to record
        """
        self.uncommitted_events.append(event)

    def get_uncommitted_events(self) -> list[DomainEvent]:
        """Get events not yet committed to event store."""
        return self.uncommitted_events.copy()

    def mark_events_as_committed(self) -> None:
        """Clear uncommitted events after committing to store."""
        self.uncommitted_events.clear()

    @abstractmethod
    def to_snapshot(self) -> dict[str, Any]:
        """
        Create snapshot of current state.

        Returns:
            Dictionary representing aggregate state
        """

    @abstractmethod
    async def from_snapshot(self, state: dict[str, Any]) -> None:
        """
        Restore aggregate from snapshot.

        Args:
            state: Snapshot state dictionary
        """


class EventSourcedRepository:
    """
    Repository for event-sourced aggregates.

    Handles saving and loading aggregates using event store and snapshots.
    """

    def __init__(
        self, event_store: EventStore, snapshot_store: SnapshotStore | None = None
    ):
        """
        Initialize repository.

        Args:
            event_store: Event store instance
            snapshot_store: Optional snapshot store for optimization
        """
        self.event_store = event_store
        self.snapshot_store = snapshot_store or SnapshotStore()

    async def save(self, aggregate: AggregateRoot) -> None:
        """
        Save aggregate by appending uncommitted events.

        Args:
            aggregate: Aggregate to save
        """
        events = aggregate.get_uncommitted_events()

        for event in events:
            await self.event_store.append(event)

        aggregate.mark_events_as_committed()

        # Create snapshot if needed
        if self.snapshot_store.should_snapshot(aggregate.version):
            snapshot = Snapshot(
                aggregate_id=aggregate.aggregate_id,
                aggregate_type=aggregate.aggregate_type,
                version=aggregate.version,
                state=aggregate.to_snapshot(),
            )
            await self.snapshot_store.save_snapshot(snapshot)

        logger.info(
            "Aggregate saved",
            aggregate_id=str(aggregate.aggregate_id),
            event_count=len(events),
            version=aggregate.version,
        )

    async def load(
        self, aggregate_type: type[AggregateRoot], aggregate_id: UUID
    ) -> AggregateRoot | None:
        """
        Load aggregate from event store.

        Args:
            aggregate_type: Type of aggregate to load
            aggregate_id: Aggregate ID

        Returns:
            Loaded aggregate or None if not found
        """
        # Try to load from snapshot first
        snapshot = await self.snapshot_store.get_snapshot(aggregate_id)

        aggregate = aggregate_type(
            aggregate_id=aggregate_id, aggregate_type=aggregate_type.__name__
        )

        if snapshot:
            # Restore from snapshot
            await aggregate.from_snapshot(snapshot.state)
            aggregate.version = snapshot.version

            # Load events after snapshot
            events = await self.event_store.get_events_for_aggregate(
                aggregate_id, from_version=snapshot.version + 1
            )
        else:
            # Load all events
            events = await self.event_store.get_events_for_aggregate(aggregate_id)

        if not events and not snapshot:
            return None

        # Replay events
        await aggregate.load_from_history(events)

        return aggregate


# Example: Enrollment Aggregate
class EnrollmentAggregate(AggregateRoot):
    """Event-sourced enrollment aggregate."""

    def __init__(self, aggregate_id: UUID, aggregate_type: str = "enrollment"):
        """Initialize enrollment aggregate."""
        super().__init__(aggregate_id, aggregate_type)
        self.student_id: UUID | None = None
        self.section_id: UUID | None = None
        self.course_id: UUID | None = None
        self.semester: str | None = None
        self.enrolled_at: datetime | None = None
        self.is_dropped: bool = False
        self.is_waitlisted: bool = False

    async def apply_event(self, event: DomainEvent) -> None:
        """Apply event to update state."""
        if isinstance(event, EnrollmentCreatedEvent):
            self.student_id = event.student_id
            self.section_id = event.section_id
            self.course_id = event.course_id
            self.semester = event.semester
            self.enrolled_at = event.enrolled_at
            self.is_waitlisted = event.is_waitlisted

        elif isinstance(event, EnrollmentDroppedEvent):
            self.is_dropped = True

    def to_snapshot(self) -> dict[str, Any]:
        """Create snapshot."""
        return {
            "student_id": str(self.student_id) if self.student_id else None,
            "section_id": str(self.section_id) if self.section_id else None,
            "course_id": str(self.course_id) if self.course_id else None,
            "semester": self.semester,
            "enrolled_at": self.enrolled_at.isoformat() if self.enrolled_at else None,
            "is_dropped": self.is_dropped,
            "is_waitlisted": self.is_waitlisted,
        }

    async def from_snapshot(self, state: dict[str, Any]) -> None:
        """Restore from snapshot."""
        self.student_id = UUID(state["student_id"]) if state.get("student_id") else None
        self.section_id = UUID(state["section_id"]) if state.get("section_id") else None
        self.course_id = UUID(state["course_id"]) if state.get("course_id") else None
        self.semester = state.get("semester")
        self.enrolled_at = (
            datetime.fromisoformat(state["enrolled_at"])
            if state.get("enrolled_at")
            else None
        )
        self.is_dropped = state.get("is_dropped", False)
        self.is_waitlisted = state.get("is_waitlisted", False)

