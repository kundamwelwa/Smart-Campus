"""
Base Event Classes

Foundation for event sourcing and event-driven architecture.
All domain events inherit from these base classes.
"""

from abc import ABC
from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class EventMetadata(BaseModel):
    """
    Event metadata for tracking and tracing.

    Provides correlation IDs, causation tracking, and distributed tracing support.
    """

    model_config = ConfigDict(frozen=True)

    event_id: UUID = Field(default_factory=uuid4, description="Unique event ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event occurrence time")
    correlation_id: UUID = Field(
        default_factory=uuid4, description="Correlation ID for request tracking"
    )
    causation_id: UUID | None = Field(
        default=None, description="ID of event that caused this event"
    )
    user_id: UUID | None = Field(default=None, description="User who triggered the event")
    service: str = Field(..., description="Originating service name")
    version: int = Field(default=1, description="Event schema version")

    # Distributed tracing
    trace_id: str | None = Field(default=None, description="Distributed trace ID")
    span_id: str | None = Field(default=None, description="Trace span ID")

    # Audit
    ip_address: str | None = Field(default=None)
    user_agent: str | None = Field(default=None)


class Event(BaseModel, ABC):
    """
    Abstract base event class.

    All events in the system inherit from this class.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    metadata: EventMetadata = Field(...)

    EVENT_TYPE: ClassVar[str] = "base.event"

    @classmethod
    def get_event_type(cls) -> str:
        """Get the event type identifier."""
        return cls.EVENT_TYPE

    def get_aggregate_id(self) -> UUID | None:
        """
        Get the aggregate root ID this event belongs to.

        Subclasses should override to provide specific aggregate ID.

        Returns:
            UUID: Aggregate root ID or None
        """
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_type": self.get_event_type(),
            "metadata": self.metadata.model_dump(),
            "payload": self.model_dump(exclude={"metadata"}),
        }


class DomainEvent(Event, ABC):
    """
    Domain event representing a significant business occurrence.

    Domain events are facts about things that have happened in the domain.
    They are immutable and form the backbone of event sourcing.
    """

    aggregate_id: UUID = Field(..., description="ID of aggregate root")
    aggregate_type: str = Field(..., description="Type of aggregate (e.g., 'Student', 'Course')")
    sequence_number: int = Field(..., ge=0, description="Event sequence in aggregate stream")

    def get_aggregate_id(self) -> UUID | None:
        """Get the aggregate root ID."""
        return self.aggregate_id


class EventEnvelope(BaseModel):
    """
    Envelope for event storage and transport.

    Wraps events with additional metadata for persistence and messaging.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4, description="Envelope ID")
    event_type: str = Field(..., description="Fully qualified event type")
    event_data: dict[str, Any] = Field(..., description="Serialized event payload")
    metadata: dict[str, Any] = Field(..., description="Event metadata")

    # Storage metadata
    stored_at: datetime = Field(default_factory=datetime.utcnow)
    stream_id: str = Field(..., description="Event stream identifier")
    stream_position: int = Field(..., ge=0, description="Position in stream")

    # Partitioning for scalability
    partition_key: str = Field(..., description="Partition key for distribution")

    @classmethod
    def wrap(
        cls,
        event: Event,
        stream_id: str,
        stream_position: int,
        partition_key: str | None = None,
    ) -> "EventEnvelope":
        """
        Wrap an event in an envelope for storage/transport.

        Args:
            event: Event to wrap
            stream_id: Event stream identifier
            stream_position: Position in stream
            partition_key: Optional partition key (defaults to aggregate_id)

        Returns:
            EventEnvelope: Wrapped event
        """
        event_dict = event.to_dict()

        return cls(
            event_type=event.get_event_type(),
            event_data=event_dict,
            metadata=event.metadata.model_dump(),
            stream_id=stream_id,
            stream_position=stream_position,
            partition_key=partition_key or str(event.get_aggregate_id() or uuid4()),
        )


class Snapshot(BaseModel):
    """
    Aggregate snapshot for optimizing event replay.

    Stores the current state of an aggregate to avoid replaying all events.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    aggregate_id: UUID = Field(..., description="Aggregate root ID")
    aggregate_type: str = Field(..., description="Aggregate type")
    state: dict[str, Any] = Field(..., description="Serialized aggregate state")
    version: int = Field(..., ge=0, description="Snapshot version (last event sequence)")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Snapshot metadata
    event_count: int = Field(..., ge=0, description="Number of events up to this snapshot")
    checksum: str | None = Field(default=None, description="State checksum for validation")

