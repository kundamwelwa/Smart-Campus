"""
Aggregate Root Base Class

Foundation for event-sourced aggregates implementing Domain-Driven Design patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from uuid import UUID

import structlog

from shared.events.base import DomainEvent

logger = structlog.get_logger(__name__)

TEvent = TypeVar("TEvent", bound=DomainEvent)


class AggregateRoot(ABC, Generic[TEvent]):
    """
    Abstract aggregate root for event sourcing.

    Aggregates:
    - Maintain their state by applying events
    - Generate domain events for state changes
    - Can be rebuilt from event stream (replay)
    - Enforce business invariants
    """

    def __init__(self, aggregate_id: UUID):
        """
        Initialize aggregate.

        Args:
            aggregate_id: Unique identifier for this aggregate
        """
        self.id = aggregate_id
        self.version = 0
        self.uncommitted_events: list[DomainEvent] = []

    @abstractmethod
    def apply_event(self, event: TEvent) -> None:
        """
        Apply an event to update aggregate state.

        This method should be pure - only update state based on event data.
        No side effects or business logic validation.

        Args:
            event: Domain event to apply
        """

    @classmethod
    @abstractmethod
    def aggregate_type(cls) -> str:
        """
        Get aggregate type identifier.

        Returns:
            str: Aggregate type name
        """

    def raise_event(self, event: TEvent) -> None:
        """
        Raise a new domain event.

        Applies the event to current state and adds to uncommitted events.

        Args:
            event: Domain event to raise
        """
        # Apply event to self
        self.apply_event(event)

        # Add to uncommitted events
        self.uncommitted_events.append(event)

        # Increment version
        self.version += 1

        logger.debug(
            "Event raised",
            aggregate_id=str(self.id),
            event_type=event.get_event_type(),
            version=self.version,
        )

    def get_uncommitted_events(self) -> list[DomainEvent]:
        """
        Get events that haven't been committed to event store.

        Returns:
            list: Uncommitted domain events
        """
        return self.uncommitted_events.copy()

    def mark_events_committed(self) -> None:
        """
        Mark all uncommitted events as committed.

        Called after events are successfully persisted to event store.
        """
        event_count = len(self.uncommitted_events)
        self.uncommitted_events.clear()

        logger.debug(
            "Events marked as committed",
            aggregate_id=str(self.id),
            count=event_count,
            version=self.version,
        )

    @classmethod
    def replay(cls, aggregate_id: UUID, events: list[TEvent]) -> "AggregateRoot[TEvent]":
        """
        Rebuild aggregate from event stream.

        Creates a new aggregate instance and applies all events in order.

        Args:
            aggregate_id: Aggregate UUID
            events: Ordered list of domain events

        Returns:
            Rebuilt aggregate instance
        """
        # Create new instance
        aggregate = cls(aggregate_id)

        # Apply all events
        for event in events:
            aggregate.apply_event(event)
            aggregate.version += 1

        logger.info(
            "Aggregate replayed from events",
            aggregate_id=str(aggregate_id),
            events_count=len(events),
            final_version=aggregate.version,
        )

        return aggregate

    def get_state(self) -> dict[str, Any]:
        """
        Get current aggregate state for snapshotting.

        Subclasses should override to provide complete state serialization.

        Returns:
            dict: Serialized state
        """
        return {
            "aggregate_id": str(self.id),
            "version": self.version,
        }

    @classmethod
    def from_snapshot(cls, snapshot_data: dict[str, Any]) -> "AggregateRoot[TEvent]":
        """
        Restore aggregate from snapshot.

        Subclasses should override to restore from snapshot data.

        Args:
            snapshot_data: Snapshot state dictionary

        Returns:
            Restored aggregate instance
        """
        aggregate_id = UUID(snapshot_data["aggregate_id"])
        aggregate = cls(aggregate_id)
        aggregate.version = snapshot_data["version"]
        return aggregate

