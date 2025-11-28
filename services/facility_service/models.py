"""
Facility Service Database Models

SQLAlchemy models for rooms, facilities, and bookings.
"""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column

from shared.database.postgres import Base


class FacilityModel(Base):
    """Facility/Building database model."""

    __tablename__ = "facilities"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    code: Mapped[str] = mapped_column(String(20), nullable=False, unique=True, index=True)
    facility_type: Mapped[str] = mapped_column(String(50), nullable=False)
    total_rooms: Mapped[int] = mapped_column(Integer, default=0)
    is_operational: Mapped[bool] = mapped_column(Boolean, default=True)
    current_temperature: Mapped[float] = mapped_column(Float, default=22.0)
    current_energy_usage: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class RoomModel(Base):
    """Room database model."""

    __tablename__ = "rooms"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    facility_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    room_number: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    room_type: Mapped[str] = mapped_column(String(50), nullable=False)
    building: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    floor: Mapped[int] = mapped_column(Integer, nullable=False)

    # Capacity
    capacity: Mapped[int] = mapped_column(Integer, nullable=False)
    area_sqm: Mapped[float] = mapped_column(Float, nullable=False)

    # Equipment
    has_projector: Mapped[bool] = mapped_column(Boolean, default=False)
    has_whiteboard: Mapped[bool] = mapped_column(Boolean, default=True)
    has_computers: Mapped[bool] = mapped_column(Boolean, default=False)
    computer_count: Mapped[int] = mapped_column(Integer, default=0)
    has_video_conference: Mapped[bool] = mapped_column(Boolean, default=False)

    # Availability
    is_available: Mapped[bool] = mapped_column(Boolean, default=True)
    is_bookable: Mapped[bool] = mapped_column(Boolean, default=True)

    # Environmental
    temperature: Mapped[float] = mapped_column(Float, default=22.0)
    current_occupancy: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class BookingModel(Base):
    """Booking database model."""

    __tablename__ = "bookings"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    room_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("rooms.id"), nullable=False, index=True
    )
    booked_by: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False, index=True)
    purpose: Mapped[str] = mapped_column(Text, nullable=False)

    # Timing
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    end_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)

    # Details
    expected_attendees: Mapped[int] = mapped_column(Integer, nullable=False)
    organizer_email: Mapped[str] = mapped_column(String(255), nullable=False)

    # Status
    is_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    is_cancelled: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

