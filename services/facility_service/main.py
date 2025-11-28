"""
Facility Service - Room Booking and Management

Handles campus facilities, room booking, and resource management.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import UUID

import structlog
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.facility_service.api import admin, rooms
from services.facility_service.models import RoomModel
from shared.config import settings
from shared.database import close_db, get_db, init_db

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan."""
    logger.info("Starting Facility Service")
    await init_db()
    yield
    await close_db()
    logger.info("Facility Service shutdown complete")


app = FastAPI(
    title="Argos Facility Service",
    description="Room booking and facility management",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
)

# Include routers
app.include_router(rooms.router, prefix="/api/v1/facilities")
app.include_router(admin.router, prefix="/api/v1", tags=["Admin"])


class RoomResponse(BaseModel):
    """Room information response."""

    id: UUID
    room_number: str
    room_type: str
    capacity: int
    building: str
    floor: int
    has_projector: bool
    has_computers: bool
    is_available: bool


class BookingRequest(BaseModel):
    """Room booking request."""

    room_id: UUID
    start_time: datetime
    end_time: datetime
    purpose: str
    expected_attendees: int


class BookingResponse(BaseModel):
    """Booking response."""

    id: UUID
    room_id: UUID
    room_number: str
    start_time: datetime
    end_time: datetime
    purpose: str
    is_confirmed: bool


@app.get("/api/v1/rooms", response_model=list[RoomResponse])
async def list_rooms(
    building: str | None = None,
    min_capacity: int | None = None,
    available_only: bool = False,
    db: AsyncSession = Depends(get_db),
) -> list[RoomResponse]:
    """
    List available rooms.

    Args:
        building: Filter by building
        min_capacity: Minimum capacity required
        available_only: Show only available rooms
        db: Database session

    Returns:
        List of rooms
    """
    query = select(RoomModel)

    if building:
        query = query.where(RoomModel.building == building)

    if min_capacity:
        query = query.where(RoomModel.capacity >= min_capacity)

    if available_only:
        query = query.where(RoomModel.is_available)

    result = await db.execute(query)
    result.scalars().all()

    # Return simplified response (models would need to be created)
    return []  # Placeholder - full implementation would return actual rooms


@app.post("/api/v1/bookings", response_model=BookingResponse, status_code=status.HTTP_201_CREATED)
async def create_booking(
    request: BookingRequest,
    db: AsyncSession = Depends(get_db),
) -> BookingResponse:
    """
    Create a room booking.

    Args:
        request: Booking request
        db: Database session

    Returns:
        Created booking
    """
    logger.info("Creating booking", room_id=str(request.room_id))

    # TODO: Implement booking logic with conflict detection
    # - Check room availability
    # - Detect time conflicts
    # - Create booking record

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Booking creation will be fully implemented",
    )


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check."""
    return {
        "status": "healthy",
        "service": "facility_service",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.facility_service.main:app",
        host="0.0.0.0",
        port=settings.facility_service_port,
        reload=settings.debug,
    )

