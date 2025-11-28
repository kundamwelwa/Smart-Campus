"""
Event Stream for Publish/Subscribe

Implements event streaming with pub/sub pattern for event-driven architecture.
Supports multiple subscribers, filtering, and event replay.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import TypeVar

import structlog

from shared.events.base import Event

logger = structlog.get_logger(__name__)

TEvent = TypeVar("TEvent", bound=Event)


class EventSubscriber(ABC):
    """
    Abstract base class for event subscribers.

    Subscribers implement this interface to receive events from streams.
    """

    @abstractmethod
    async def handle_event(self, event: Event) -> None:
        """
        Handle an event.

        Args:
            event: Event to handle
        """

    @abstractmethod
    def get_subscribed_event_types(self) -> list[type[Event]]:
        """
        Get list of event types this subscriber is interested in.

        Returns:
            List of event type classes
        """


class EventStream:
    """
    Event stream for publish/subscribe pattern.

    Supports:
    - Multiple subscribers per event type
    - Event filtering
    - Event replay
    - Async event handling
    """

    def __init__(self, stream_id: str):
        """
        Initialize event stream.

        Args:
            stream_id: Unique stream identifier
        """
        self.stream_id = stream_id
        self._subscribers: dict[type[Event], list[EventSubscriber]] = defaultdict(list)
        self._event_history: list[Event] = []
        self._max_history: int | None = None  # None = unlimited
        logger.info("Event stream created", stream_id=stream_id)

    def subscribe(
        self, event_type: type[TEvent], subscriber: EventSubscriber
    ) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Event type class to subscribe to
            subscriber: Subscriber instance
        """
        if subscriber not in self._subscribers[event_type]:
            self._subscribers[event_type].append(subscriber)
            logger.info(
                "Subscriber registered",
                stream_id=self.stream_id,
                event_type=event_type.__name__,
                subscriber=subscriber.__class__.__name__,
            )

    def unsubscribe(
        self, event_type: type[TEvent], subscriber: EventSubscriber
    ) -> None:
        """
        Unsubscribe from events of a specific type.

        Args:
            event_type: Event type class
            subscriber: Subscriber to remove
        """
        if subscriber in self._subscribers[event_type]:
            self._subscribers[event_type].remove(subscriber)
            logger.info(
                "Subscriber unregistered",
                stream_id=self.stream_id,
                event_type=event_type.__name__,
            )

    async def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        # Add to history
        self._event_history.append(event)

        # Enforce history limit
        if self._max_history is not None and len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Find all subscribers for this event type and its base classes
        event_type = type(event)
        subscribers_to_notify: list[EventSubscriber] = []

        # Get subscribers for exact type
        subscribers_to_notify.extend(self._subscribers.get(event_type, []))

        # Get subscribers for base classes (polymorphic subscription)
        for base_type in event_type.__mro__:
            if base_type != event_type and issubclass(base_type, Event):
                subscribers_to_notify.extend(self._subscribers.get(base_type, []))

        # Notify all subscribers
        for subscriber in subscribers_to_notify:
            try:
                await subscriber.handle_event(event)
            except Exception as e:
                logger.error(
                    "Subscriber error",
                    stream_id=self.stream_id,
                    event_type=event_type.__name__,
                    subscriber=subscriber.__class__.__name__,
                    error=str(e),
                )

        logger.debug(
            "Event published",
            stream_id=self.stream_id,
            event_type=event_type.__name__,
            subscribers_notified=len(subscribers_to_notify),
        )

    async def replay_events(
        self,
        subscriber: EventSubscriber,
        event_type: type[Event] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> int:
        """
        Replay historical events to a subscriber.

        Args:
            subscriber: Subscriber to replay events to
            event_type: Optional event type filter
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            Number of events replayed
        """
        events_to_replay = self._event_history

        # Apply filters
        if event_type:
            events_to_replay = [e for e in events_to_replay if isinstance(e, event_type)]

        if start_time:
            events_to_replay = [
                e for e in events_to_replay if e.metadata.timestamp >= start_time
            ]

        if end_time:
            events_to_replay = [
                e for e in events_to_replay if e.metadata.timestamp <= end_time
            ]

        # Replay events
        for event in events_to_replay:
            try:
                await subscriber.handle_event(event)
            except Exception as e:
                logger.error(
                    "Replay error",
                    stream_id=self.stream_id,
                    event_type=type(event).__name__,
                    error=str(e),
                )

        logger.info(
            "Events replayed",
            stream_id=self.stream_id,
            count=len(events_to_replay),
        )

        return len(events_to_replay)

    def get_event_history(
        self,
        event_type: type[Event] | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[Event]:
        """
        Get event history with optional filters.

        Args:
            event_type: Optional event type filter
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of matching events
        """
        events = self._event_history

        if event_type:
            events = [e for e in events if isinstance(e, event_type)]

        if start_time:
            events = [e for e in events if e.metadata.timestamp >= start_time]

        if end_time:
            events = [e for e in events if e.metadata.timestamp <= end_time]

        return events

    def set_max_history(self, max_events: int | None) -> None:
        """
        Set maximum number of events to keep in history.

        Args:
            max_events: Maximum events (None = unlimited)
        """
        self._max_history = max_events
        if max_events is not None and len(self._event_history) > max_events:
            self._event_history = self._event_history[-max_events:]

    def get_subscriber_count(self, event_type: type[Event] | None = None) -> int:
        """
        Get number of subscribers.

        Args:
            event_type: Optional event type filter

        Returns:
            Total subscriber count
        """
        if event_type:
            return len(self._subscribers.get(event_type, []))
        return sum(len(subs) for subs in self._subscribers.values())

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
        logger.info("Event history cleared", stream_id=self.stream_id)


class EventStreamManager:
    """
    Manager for multiple event streams.

    Provides centralized access to streams and cross-stream operations.
    """

    def __init__(self):
        """Initialize stream manager."""
        self._streams: dict[str, EventStream] = {}

    def get_or_create_stream(self, stream_id: str) -> EventStream:
        """
        Get existing stream or create new one.

        Args:
            stream_id: Stream identifier

        Returns:
            EventStream instance
        """
        if stream_id not in self._streams:
            self._streams[stream_id] = EventStream(stream_id)
        return self._streams[stream_id]

    def get_stream(self, stream_id: str) -> EventStream | None:
        """Get stream by ID."""
        return self._streams.get(stream_id)

    def remove_stream(self, stream_id: str) -> bool:
        """
        Remove a stream.

        Args:
            stream_id: Stream identifier

        Returns:
            True if stream was found and removed
        """
        if stream_id in self._streams:
            del self._streams[stream_id]
            return True
        return False

    def list_streams(self) -> list[str]:
        """List all stream IDs."""
        return list(self._streams.keys())


# Global stream manager instance
_stream_manager: EventStreamManager | None = None


def get_event_stream_manager() -> EventStreamManager:
    """
    Get or create global event stream manager.

    Returns:
        EventStreamManager instance
    """
    global _stream_manager

    if _stream_manager is None:
        _stream_manager = EventStreamManager()

    return _stream_manager

