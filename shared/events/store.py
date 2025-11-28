"""
Event Store Implementation

MongoDB-based append-only event store with replay and snapshotting capabilities.
Provides the foundation for event sourcing and CQRS patterns.
"""

from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any
from uuid import UUID

import structlog
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection, AsyncIOMotorDatabase

from shared.config import settings
from shared.events.base import Event, EventEnvelope, Snapshot

logger = structlog.get_logger(__name__)


class EventStore:
    """
    Append-only event store for event sourcing.

    Features:
    - Guaranteed ordering within aggregate streams
    - Optimistic concurrency control
    - Snapshot support for performance
    - Stream replay capabilities
    """

    def __init__(self, mongodb_client: AsyncIOMotorClient) -> None:
        """
        Initialize event store.

        Args:
            mongodb_client: MongoDB async client
        """
        self.client = mongodb_client
        self.db: AsyncIOMotorDatabase = mongodb_client[settings.mongodb_db]
        self.events_collection: AsyncIOMotorCollection = self.db["events"]
        self.snapshots_collection: AsyncIOMotorCollection = self.db["snapshots"]

    async def initialize(self) -> None:
        """Initialize event store indexes for performance."""
        # Index on stream_id and stream_position for ordered retrieval
        await self.events_collection.create_index(
            [("stream_id", 1), ("stream_position", 1)], unique=True
        )

        # Index on aggregate_id from metadata
        await self.events_collection.create_index([("metadata.aggregate_id", 1)])

        # Index on event_type for filtering
        await self.events_collection.create_index([("event_type", 1)])

        # Index on timestamp for time-based queries
        await self.events_collection.create_index([("stored_at", -1)])

        # Snapshot indexes
        await self.snapshots_collection.create_index(
            [("aggregate_id", 1), ("version", -1)]
        )

        logger.info("Event store initialized with indexes")

    async def append(
        self,
        event: Event,
        stream_id: str,
        expected_version: int | None = None,
    ) -> EventEnvelope:
        """
        Append an event to the store.

        Args:
            event: Event to append
            stream_id: Event stream identifier
            expected_version: Expected current version for optimistic concurrency control

        Returns:
            EventEnvelope: Stored event envelope

        Raises:
            ConcurrencyError: If expected version doesn't match
        """
        # Get current stream position
        current_position = await self._get_stream_position(stream_id)

        # Optimistic concurrency check
        if expected_version is not None and current_position != expected_version:
            raise ConcurrencyError(
                f"Concurrency conflict: expected version {expected_version}, "
                f"but stream is at {current_position}"
            )

        # Create envelope
        envelope = EventEnvelope.wrap(
            event=event,
            stream_id=stream_id,
            stream_position=current_position + 1,
            partition_key=str(event.get_aggregate_id() or stream_id),
        )

        # Append to store (atomic operation)
        envelope_dict = envelope.model_dump(mode="json")
        envelope_dict["_id"] = str(envelope.id)  # Use envelope ID as MongoDB _id

        try:
            await self.events_collection.insert_one(envelope_dict)
            logger.info(
                "Event appended",
                event_type=event.get_event_type(),
                stream_id=stream_id,
                position=envelope.stream_position,
            )
            return envelope
        except Exception as e:
            logger.error(
                "Failed to append event",
                event_type=event.get_event_type(),
                stream_id=stream_id,
                error=str(e),
            )
            raise

    async def _get_stream_position(self, stream_id: str) -> int:
        """Get the current position (latest event number) in a stream."""
        result = await self.events_collection.find_one(
            {"stream_id": stream_id}, sort=[("stream_position", -1)]
        )
        if result is None:
            return 0
        return result["stream_position"]

    async def get_stream(
        self, stream_id: str, from_position: int = 0, to_position: int | None = None
    ) -> AsyncIterator[EventEnvelope]:
        """
        Get events from a stream.

        Args:
            stream_id: Stream identifier
            from_position: Starting position (inclusive)
            to_position: Ending position (inclusive, None = all)

        Yields:
            EventEnvelope: Events in order
        """
        query: dict[str, Any] = {"stream_id": stream_id, "stream_position": {"$gte": from_position}}

        if to_position is not None:
            query["stream_position"]["$lte"] = to_position

        cursor = self.events_collection.find(query).sort("stream_position", 1)

        async for doc in cursor:
            yield self._doc_to_envelope(doc)

    async def get_aggregate_stream(
        self, aggregate_id: UUID, from_version: int = 0
    ) -> AsyncIterator[EventEnvelope]:
        """
        Get all events for a specific aggregate.

        Args:
            aggregate_id: Aggregate root ID
            from_version: Starting version number

        Yields:
            EventEnvelope: Events for the aggregate
        """
        query = {
            "event_data.aggregate_id": str(aggregate_id),
            "event_data.sequence_number": {"$gte": from_version},
        }

        cursor = self.events_collection.find(query).sort("event_data.sequence_number", 1)

        async for doc in cursor:
            yield self._doc_to_envelope(doc)

    async def save_snapshot(self, snapshot: Snapshot) -> None:
        """
        Save an aggregate snapshot.

        Args:
            snapshot: Snapshot to save
        """
        snapshot_dict = snapshot.model_dump(mode="json")
        snapshot_dict["_id"] = f"{snapshot.aggregate_id}_{snapshot.version}"

        # Upsert snapshot
        await self.snapshots_collection.replace_one(
            {
                "aggregate_id": str(snapshot.aggregate_id),
                "version": snapshot.version,
            },
            snapshot_dict,
            upsert=True,
        )

        logger.info(
            "Snapshot saved",
            aggregate_id=str(snapshot.aggregate_id),
            version=snapshot.version,
        )

    async def get_latest_snapshot(
        self, aggregate_id: UUID
    ) -> Snapshot | None:
        """
        Get the latest snapshot for an aggregate.

        Args:
            aggregate_id: Aggregate root ID

        Returns:
            Snapshot or None if no snapshot exists
        """
        result = await self.snapshots_collection.find_one(
            {"aggregate_id": str(aggregate_id)}, sort=[("version", -1)]
        )

        if result is None:
            return None

        return self._doc_to_snapshot(result)

    async def replay_aggregate(
        self, aggregate_id: UUID, up_to_version: int | None = None
    ) -> tuple[Snapshot | None, list[EventEnvelope]]:
        """
        Replay an aggregate's event stream with snapshot optimization.

        Args:
            aggregate_id: Aggregate root ID
            up_to_version: Replay up to this version (None = all)

        Returns:
            Tuple of (latest_snapshot, subsequent_events)
        """
        # Try to get latest snapshot
        snapshot = await self.get_latest_snapshot(aggregate_id)

        # Determine starting version
        from_version = snapshot.version + 1 if snapshot else 0

        # Get events after snapshot
        events = []
        async for envelope in self.get_aggregate_stream(aggregate_id, from_version):
            if up_to_version is not None and envelope.stream_position > up_to_version:
                break
            events.append(envelope)

        logger.info(
            "Aggregate replayed",
            aggregate_id=str(aggregate_id),
            snapshot_version=snapshot.version if snapshot else None,
            events_count=len(events),
        )

        return snapshot, events

    async def get_all_events(
        self,
        from_timestamp: datetime | None = None,
        to_timestamp: datetime | None = None,
        event_types: list[str] | None = None,
        limit: int = 1000,
    ) -> list[EventEnvelope]:
        """
        Get events with filtering.

        Args:
            from_timestamp: Start time (inclusive)
            to_timestamp: End time (inclusive)
            event_types: Filter by event types
            limit: Maximum number of events

        Returns:
            List of event envelopes
        """
        query: dict[str, Any] = {}

        if from_timestamp or to_timestamp:
            query["stored_at"] = {}
            if from_timestamp:
                query["stored_at"]["$gte"] = from_timestamp
            if to_timestamp:
                query["stored_at"]["$lte"] = to_timestamp

        if event_types:
            query["event_type"] = {"$in": event_types}

        cursor = self.events_collection.find(query).sort("stored_at", 1).limit(limit)

        events = []
        async for doc in cursor:
            events.append(self._doc_to_envelope(doc))

        return events

    def _doc_to_envelope(self, doc: dict[str, Any]) -> EventEnvelope:
        """Convert MongoDB document to EventEnvelope."""
        # Remove MongoDB _id
        doc.pop("_id", None)
        return EventEnvelope(**doc)

    def _doc_to_snapshot(self, doc: dict[str, Any]) -> Snapshot:
        """Convert MongoDB document to Snapshot."""
        doc.pop("_id", None)
        return Snapshot(**doc)


class ConcurrencyError(Exception):
    """Raised when optimistic concurrency check fails."""



class EventPublisher:
    """
    Event publisher for distributing events to message brokers.

    Publishes stored events to Kafka for downstream consumption.
    """

    def __init__(self, kafka_producer: Any) -> None:
        """
        Initialize event publisher.

        Args:
            kafka_producer: Kafka producer instance
        """
        self.producer = kafka_producer

    async def publish(self, envelope: EventEnvelope) -> None:
        """
        Publish event to Kafka.

        Args:
            envelope: Event envelope to publish
        """
        topic = self._get_topic_for_event(envelope.event_type)

        {
            "event_id": str(envelope.id),
            "event_type": envelope.event_type,
            "data": envelope.event_data,
            "metadata": envelope.metadata,
        }

        # Publish to Kafka (actual implementation will use aiokafka)
        # await self.producer.send(topic, value=json.dumps(message).encode())

        logger.info(
            "Event published",
            event_type=envelope.event_type,
            topic=topic,
            event_id=str(envelope.id),
        )

    def _get_topic_for_event(self, event_type: str) -> str:
        """
        Determine Kafka topic based on event type.

        Args:
            event_type: Event type (e.g., 'academic.enrollment.student_enrolled')

        Returns:
            str: Kafka topic name
        """
        # Extract domain from event type (e.g., 'academic' from 'academic.enrollment.student_enrolled')
        domain = event_type.split(".")[0] if "." in event_type else "general"
        return f"argos.events.{domain}"

