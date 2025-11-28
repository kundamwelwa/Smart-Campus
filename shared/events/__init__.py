"""
Event System for Event-Driven Architecture

Provides event sourcing, publish/subscribe, and event streaming capabilities.
"""

from shared.events.base import (
    DomainEvent,
    Event,
    EventEnvelope,
    EventMetadata,
    Snapshot,
)
from shared.events.stream import (
    EventStream,
    EventStreamManager,
    EventSubscriber,
    get_event_stream_manager,
)

__all__ = [
    # Base Events
    "Event",
    "DomainEvent",
    "EventMetadata",
    "EventEnvelope",
    "Snapshot",
    # Event Streaming
    "EventStream",
    "EventStreamManager",
    "EventSubscriber",
    "get_event_stream_manager",
]
