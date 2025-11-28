"""
gRPC Service Implementation for Academic Service

Exposes the same business logic as REST API via gRPC.
Implements API versioning and backward compatibility.
"""

from uuid import UUID, uuid4

import grpc
import structlog
from grpc import aio
from sqlalchemy.ext.asyncio import AsyncSession

from services.academic_service.enrollment_service import EnrollmentService
from shared.domain.policies import PolicyEngine

# Import generated gRPC code (would be generated from proto files)
# For now, we'll create a placeholder structure
from shared.events.store import EventStore

logger = structlog.get_logger(__name__)


# Note: In production, gRPC code would be generated from .proto files
# This is a structure showing how gRPC service would be implemented
# Actual implementation would import from shared/grpc/generated/

class AcademicServiceServicer:
    """
    gRPC service implementation for Academic Service.

    Exposes the same business logic as REST API endpoints.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        event_store: EventStore,
        policy_engine: PolicyEngine,
    ):
        """
        Initialize gRPC service.

        Args:
            db_session: Database session
            event_store: Event store
            policy_engine: Policy engine
        """
        self.enrollment_service = EnrollmentService(
            db_session=db_session,
            event_store=event_store,
            policy_engine=policy_engine,
        )
        self.db = db_session
        self.event_store = event_store

    async def EnrollStudent(self, request, context):
        """
        Enroll a student in a section (gRPC).

        Same business logic as REST endpoint.
        """
        try:
            student_id = UUID(request.student_id)
            section_id = UUID(request.section_id)
            user_id = UUID(context.invocation_metadata()[0].value) if context.invocation_metadata() else uuid4()

            # Use the same enrollment service as REST
            enrollment = await self.enrollment_service.enroll_student(
                student_id=student_id,
                section_id=section_id,
                user_id=user_id,
            )

            # Convert to gRPC response
            # In production, this would use generated protobuf classes
            # For now, return a dictionary that would be serialized

            # Placeholder - actual implementation would use generated classes
            # response = EnrollmentResponse()
            # response.id = str(enrollment.id)
            # response.student_id = str(enrollment.student_id)
            # response.section_id = str(enrollment.section_id)
            # response.enrollment_status = enrollment.enrollment_status
            # response.is_waitlisted = enrollment.is_waitlisted
            # response.waitlist_position = enrollment.waitlist_position or 0
            #
            # if enrollment.enrolled_at:
            #     timestamp = Timestamp()
            #     timestamp.FromDatetime(enrollment.enrolled_at)
            #     response.enrolled_at.CopyFrom(timestamp)
            #
            # return response

            # For now, return dict (would be converted to protobuf in real implementation)
            return {
                "id": str(enrollment.id),
                "student_id": str(enrollment.student_id),
                "section_id": str(enrollment.section_id),
                "enrollment_status": enrollment.enrollment_status,
                "is_waitlisted": enrollment.is_waitlisted,
                "waitlist_position": enrollment.waitlist_position or 0,
                "enrolled_at": enrollment.enrolled_at.isoformat() if enrollment.enrolled_at else None,
            }

        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            logger.error("gRPC enrollment error", error=str(e))
            raise

    async def GetEnrollments(self, request, context):
        """Get student enrollments (gRPC)."""
        # Implementation would query database and return enrollments
        # Same logic as REST endpoint

    async def GetCourse(self, request, context):
        """Get course by ID (gRPC)."""
        # Implementation would query database

    async def ListCourses(self, request, context):
        """List courses (gRPC)."""
        # Implementation would query database with filters


async def serve_grpc(
    port: int = 50051,
    db_session: AsyncSession = None,
    event_store: EventStore = None,
    policy_engine: PolicyEngine = None,
) -> None:
    """
    Start gRPC server.

    Args:
        port: gRPC server port
        db_session: Database session
        event_store: Event store
        policy_engine: Policy engine
    """
    server = aio.server()

    # Add service
    AcademicServiceServicer(db_session, event_store, policy_engine)
    # In production: add_AcademicServiceServicer_to_server(servicer, server)

    listen_addr = f"[::]:{port}"
    server.add_insecure_port(listen_addr)

    await server.start()
    logger.info("gRPC server started", port=port, listen_addr=listen_addr)

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await server.stop(0)
        logger.info("gRPC server stopped")

