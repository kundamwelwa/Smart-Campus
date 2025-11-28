"""
Scheduler Service - Enrollment Integration

Demonstrates concurrent service communication between EnrollmentService and SchedulerService.
Services communicate via events and HTTP/gRPC.
"""

import httpx
import structlog

from shared.events.academic_events import SectionCreatedEvent, StudentEnrolledEvent
from shared.events.stream import EventSubscriber, get_event_stream_manager

logger = structlog.get_logger(__name__)


class EnrollmentEventSubscriber(EventSubscriber):
    """
    Scheduler service subscriber for enrollment events.

    Listens to enrollment events and updates timetable accordingly.
    """

    def __init__(self, scheduler_service_url: str):
        """
        Initialize subscriber.

        Args:
            scheduler_service_url: Base URL of scheduler service
        """
        self.scheduler_service_url = scheduler_service_url
        self.http_client = httpx.AsyncClient()

    async def handle_event(self, event) -> None:
        """Handle enrollment events."""
        if isinstance(event, StudentEnrolledEvent):
            await self._handle_student_enrolled(event)
        elif isinstance(event, SectionCreatedEvent):
            await self._handle_section_created(event)

    async def _handle_student_enrolled(self, event: StudentEnrolledEvent) -> None:
        """Handle student enrollment event."""
        logger.info(
            "Processing enrollment event in scheduler",
            student_id=str(event.student_id),
            section_id=str(event.section_id),
        )

        # Notify scheduler service about enrollment
        # This could trigger timetable recalculation if needed
        try:
            response = await self.http_client.post(
                f"{self.scheduler_service_url}/api/v1/enrollments/notify",
                json={
                    "student_id": str(event.student_id),
                    "section_id": str(event.section_id),
                    "enrolled_at": event.enrolled_at.isoformat(),
                },
            )
            response.raise_for_status()
        except Exception as e:
            logger.error("Failed to notify scheduler service", error=str(e))

    async def _handle_section_created(self, event: SectionCreatedEvent) -> None:
        """Handle section creation event."""
        logger.info(
            "Processing section creation in scheduler",
            section_id=str(event.section_id),
            course_code=event.course_code,
        )

        # Request timetable generation for new section
        try:
            response = await self.http_client.post(
                f"{self.scheduler_service_url}/api/v1/generate",
                json={
                    "sections": [{"section_id": str(event.section_id)}],
                    "action": "add_section",
                },
            )
            response.raise_for_status()
        except Exception as e:
            logger.error("Failed to request timetable update", error=str(e))

    def get_subscribed_event_types(self) -> list:
        """Get subscribed event types."""
        from shared.events.academic_events import (
            SectionCreatedEvent,
            StudentEnrolledEvent,
        )
        return [StudentEnrolledEvent, SectionCreatedEvent]


async def setup_scheduler_enrollment_integration(
    scheduler_service_url: str = "http://localhost:8003",
    event_stream_id: str = "academic_events",
) -> None:
    """
    Set up integration between enrollment and scheduler services.

    Args:
        scheduler_service_url: Scheduler service URL
        event_stream_id: Event stream identifier
    """
    stream_manager = get_event_stream_manager()
    event_stream = stream_manager.get_or_create_stream(event_stream_id)

    # Create and register subscriber
    subscriber = EnrollmentEventSubscriber(scheduler_service_url)

    # Subscribe to enrollment events
    from shared.events.academic_events import SectionCreatedEvent, StudentEnrolledEvent

    event_stream.subscribe(StudentEnrolledEvent, subscriber)
    event_stream.subscribe(SectionCreatedEvent, subscriber)

    logger.info(
        "Scheduler-Enrollment integration established",
        scheduler_url=scheduler_service_url,
        stream_id=event_stream_id,
    )

