"""
Facilities Domain Models

Entities related to campus facilities, rooms, resources, and IoT sensors/actuators.
"""

from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator

from shared.domain.entities import VersionedEntity


class FacilityType(str, Enum):
    """Type of campus facility."""

    ACADEMIC_BUILDING = "academic_building"
    LIBRARY = "library"
    LABORATORY = "laboratory"
    SPORTS_COMPLEX = "sports_complex"
    DORMITORY = "dormitory"
    CAFETERIA = "cafeteria"
    AUDITORIUM = "auditorium"
    PARKING = "parking"
    ADMINISTRATIVE = "administrative"


class RoomType(str, Enum):
    """Type of room within a facility."""

    CLASSROOM = "classroom"
    LECTURE_HALL = "lecture_hall"
    COMPUTER_LAB = "computer_lab"
    RESEARCH_LAB = "research_lab"
    OFFICE = "office"
    MEETING_ROOM = "meeting_room"
    STUDY_ROOM = "study_room"
    WORKSHOP = "workshop"
    STORAGE = "storage"


class ResourceType(str, Enum):
    """Type of room resource."""

    PROJECTOR = "projector"
    WHITEBOARD = "whiteboard"
    COMPUTER = "computer"
    SMART_BOARD = "smart_board"
    VIDEO_CONFERENCE = "video_conference"
    AUDIO_SYSTEM = "audio_system"
    LAB_EQUIPMENT = "lab_equipment"


class Facility(VersionedEntity):
    """
    Campus facility/building entity.

    Represents physical buildings with associated metadata and energy monitoring.
    """

    name: str = Field(..., min_length=1, max_length=200)
    code: str = Field(..., description="Building code (e.g., 'SCI-A')")
    facility_type: FacilityType = Field(...)
    address: str = Field(..., max_length=500)
    total_floors: int = Field(..., ge=1, le=100)
    total_rooms: int = Field(default=0, ge=0)

    # Capacity
    total_capacity: int = Field(default=0, ge=0, description="Total occupancy capacity")
    accessible: bool = Field(default=True, description="ADA accessible")

    # Energy & Environment
    energy_meter_id: str | None = Field(default=None)
    current_energy_usage: float = Field(default=0.0, ge=0.0, description="Current kWh usage")
    target_temperature: float = Field(default=22.0, ge=10.0, le=30.0, description="Celsius")
    current_temperature: float = Field(default=22.0, ge=-10.0, le=50.0)

    # Operational
    is_operational: bool = Field(default=True)
    maintenance_required: bool = Field(default=False)
    last_maintenance_date: date | None = Field(default=None)

    @field_validator("code")
    @classmethod
    def validate_code(cls, v: str) -> str:
        """Validate facility code format."""
        if not v or len(v) < 2:
            raise ValueError("Facility code must be at least 2 characters")
        return v.upper()

    def validate_business_rules(self) -> bool:
        """Validate facility business rules."""
        if self.total_floors < 1:
            raise ValueError("Facility must have at least one floor")
        if self.total_rooms < 0:
            raise ValueError("Total rooms cannot be negative")
        return True


class Room(VersionedEntity):
    """
    Room entity within a facility.

    Represents individual rooms with capacity, equipment, and scheduling.
    """

    facility_id: UUID = Field(..., description="Parent facility ID")
    room_number: str = Field(..., description="Room number/identifier")
    room_type: RoomType = Field(...)
    floor: int = Field(..., ge=0, description="Floor number (0 = ground)")

    # Capacity
    capacity: int = Field(..., ge=1, le=1000, description="Maximum occupancy")
    current_occupancy: int = Field(default=0, ge=0)

    # Dimensions
    area_sqm: float = Field(..., gt=0.0, description="Floor area in square meters")

    # Features & Equipment
    has_projector: bool = Field(default=False)
    has_whiteboard: bool = Field(default=False)
    has_computers: bool = Field(default=False)
    computer_count: int = Field(default=0, ge=0)
    has_video_conference: bool = Field(default=False)
    has_audio_system: bool = Field(default=False)
    accessible: bool = Field(default=True, description="ADA accessible")

    # Availability
    is_bookable: bool = Field(default=True)
    is_available: bool = Field(default=True)
    requires_approval: bool = Field(default=False)

    # Environmental
    temperature: float = Field(default=22.0, ge=-10.0, le=50.0)
    humidity: float = Field(default=50.0, ge=0.0, le=100.0)
    light_level: float = Field(default=500.0, ge=0.0, description="Lux")

    def validate_business_rules(self) -> bool:
        """Validate room business rules."""
        if self.current_occupancy > self.capacity:
            raise ValueError("Current occupancy exceeds capacity")
        if self.computer_count > 0 and not self.has_computers:
            raise ValueError("Computer count set but has_computers is False")
        return True

    def is_occupied(self) -> bool:
        """Check if room is currently occupied."""
        return self.current_occupancy > 0

    def can_accommodate(self, required_capacity: int) -> bool:
        """Check if room can accommodate required capacity."""
        return self.capacity >= required_capacity and self.is_available


class Resource(VersionedEntity):
    """
    Room resource/equipment entity.

    Represents individual pieces of equipment or resources.
    """

    room_id: UUID = Field(..., description="Parent room ID")
    resource_type: ResourceType = Field(...)
    name: str = Field(..., max_length=200)
    manufacturer: str | None = Field(default=None, max_length=100)
    model: str | None = Field(default=None, max_length=100)
    serial_number: str | None = Field(default=None, max_length=100)

    # Status
    is_operational: bool = Field(default=True)
    requires_maintenance: bool = Field(default=False)
    last_maintenance_date: date | None = Field(default=None)
    next_maintenance_date: date | None = Field(default=None)

    # Usage tracking
    usage_hours: float = Field(default=0.0, ge=0.0)
    last_used_at: datetime | None = Field(default=None)

    def validate_business_rules(self) -> bool:
        """Validate resource business rules."""
        return True

    def record_usage(self, hours: float) -> None:
        """Record resource usage."""
        self.usage_hours += hours
        self.last_used_at = datetime.utcnow()
        self.mark_updated()


class Sensor(VersionedEntity):
    """
    IoT sensor entity for environmental monitoring.

    Represents physical or virtual sensors collecting real-time data.
    """

    room_id: UUID | None = Field(default=None, description="Associated room ID")
    facility_id: UUID | None = Field(default=None, description="Associated facility ID")

    sensor_type: str = Field(..., description="Type (temperature, humidity, motion, energy, etc.)")
    manufacturer: str | None = Field(default=None, max_length=100)
    model: str | None = Field(default=None, max_length=100)
    hardware_id: str = Field(..., description="Hardware/MAC address")

    # Current Reading
    current_value: float = Field(default=0.0)
    unit: str = Field(..., max_length=20, description="Unit of measurement")
    last_reading_at: datetime = Field(default_factory=datetime.utcnow)

    # Calibration & Status
    is_online: bool = Field(default=True)
    is_calibrated: bool = Field(default=True)
    last_calibration_date: date | None = Field(default=None)
    battery_level: float | None = Field(default=None, ge=0.0, le=100.0)

    # Thresholds for anomaly detection
    min_threshold: float | None = Field(default=None)
    max_threshold: float | None = Field(default=None)

    def validate_business_rules(self) -> bool:
        """Validate sensor business rules."""
        if (
            self.min_threshold is not None
            and self.max_threshold is not None
            and self.min_threshold >= self.max_threshold
        ):
            raise ValueError("Min threshold must be less than max threshold")
        return True

    def update_reading(self, value: float) -> None:
        """Update sensor reading."""
        self.current_value = value
        self.last_reading_at = datetime.utcnow()
        self.mark_updated()

    def is_reading_anomalous(self) -> bool:
        """Check if current reading is outside thresholds."""
        if self.min_threshold is not None and self.current_value < self.min_threshold:
            return True
        return bool(self.max_threshold is not None and self.current_value > self.max_threshold)


class Actuator(VersionedEntity):
    """
    IoT actuator entity for environmental control.

    Represents devices that can be controlled (lights, HVAC, locks, etc.).
    """

    room_id: UUID | None = Field(default=None, description="Associated room ID")
    facility_id: UUID | None = Field(default=None, description="Associated facility ID")

    actuator_type: str = Field(
        ..., description="Type (hvac, light, lock, blind, power_outlet, etc.)"
    )
    manufacturer: str | None = Field(default=None, max_length=100)
    model: str | None = Field(default=None, max_length=100)
    hardware_id: str = Field(..., description="Hardware/MAC address")

    # Current State
    is_online: bool = Field(default=True)
    current_state: dict[str, Any] = Field(
        default_factory=dict, description="Current actuator state"
    )
    last_command_at: datetime | None = Field(default=None)
    last_command_by: UUID | None = Field(default=None, description="User who last controlled")

    # Safety
    requires_authorization: bool = Field(default=False)
    authorized_users: list[UUID] = Field(default_factory=list)

    def validate_business_rules(self) -> bool:
        """Validate actuator business rules."""
        return True

    def send_command(self, command: dict[str, Any], user_id: UUID) -> None:
        """
        Send command to actuator.

        Args:
            command: Command dictionary
            user_id: User issuing the command

        Raises:
            PermissionError: If user not authorized
        """
        if self.requires_authorization and user_id not in self.authorized_users:
            raise PermissionError("User not authorized to control this actuator")

        self.current_state.update(command)
        self.last_command_at = datetime.utcnow()
        self.last_command_by = user_id
        self.mark_updated()


class Booking(VersionedEntity):
    """
    Room booking/reservation entity.

    Manages room reservations with conflict detection.
    """

    room_id: UUID = Field(..., description="Booked room ID")
    booked_by: UUID = Field(..., description="User who made booking")
    purpose: str = Field(..., max_length=500)

    # Timing
    start_time: datetime = Field(...)
    end_time: datetime = Field(...)

    # Attendees
    expected_attendees: int = Field(..., ge=1, description="Expected number of attendees")
    organizer_email: str = Field(...)

    # Status
    is_confirmed: bool = Field(default=False)
    is_cancelled: bool = Field(default=False)
    requires_setup: bool = Field(default=False)
    setup_notes: str | None = Field(default=None, max_length=1000)

    def validate_business_rules(self) -> bool:
        """Validate booking business rules."""
        if self.end_time <= self.start_time:
            raise ValueError("End time must be after start time")
        if self.is_cancelled and self.is_confirmed:
            raise ValueError("Cannot be both confirmed and cancelled")
        return True

    def confirm(self) -> None:
        """Confirm the booking."""
        if self.is_cancelled:
            raise ValueError("Cannot confirm a cancelled booking")
        self.is_confirmed = True
        self.mark_updated()

    def cancel(self) -> None:
        """Cancel the booking."""
        self.is_cancelled = True
        self.is_confirmed = False
        self.mark_updated()

    def overlaps_with(self, other_start: datetime, other_end: datetime) -> bool:
        """
        Check if this booking overlaps with another time range.

        Args:
            other_start: Other booking start time
            other_end: Other booking end time

        Returns:
            bool: True if there's an overlap
        """
        return self.start_time < other_end and other_start < self.end_time

