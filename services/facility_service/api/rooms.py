"""
Room and Facility API endpoints.
"""

from datetime import datetime
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.facility_service.models import FacilityModel, RoomModel
from shared.database import get_db

router = APIRouter(prefix="/rooms", tags=["rooms"])
logger = structlog.get_logger(__name__)


@router.get("")
async def list_rooms(
    facility_id: UUID | None = None,
    is_available: bool | None = None,
    min_capacity: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
    List all rooms with optional filters.

    Args:
        facility_id: Filter by facility
        is_available: Filter by availability
        min_capacity: Minimum capacity required

    Returns:
        List of rooms with facility information
    """
    try:
        # Build query
        stmt = (
            select(RoomModel, FacilityModel)
            .join(FacilityModel, RoomModel.facility_id == FacilityModel.id)
        )

        if facility_id:
            stmt = stmt.where(RoomModel.facility_id == facility_id)
        if is_available is not None:
            stmt = stmt.where(RoomModel.is_available == is_available)
        if min_capacity:
            stmt = stmt.where(RoomModel.capacity >= min_capacity)

        result = await db.execute(stmt)
        rows = result.all()

        rooms = []
        for room_model, facility_model in rows:
            rooms.append({
                "id": str(room_model.id),
                "facility_id": str(facility_model.id),
                "facility_name": facility_model.name,
                "room_number": room_model.room_number,
                "room_type": room_model.room_type,
                "floor": room_model.floor,
                "capacity": room_model.capacity,
                "current_occupancy": room_model.current_occupancy,
                "is_available": room_model.is_available,
                "has_projector": room_model.has_projector,
                "has_whiteboard": room_model.has_whiteboard,
                "has_computers": room_model.has_computers,
                "has_wifi": True,  # Assume all rooms have WiFi
                "has_video_conference": room_model.has_video_conference,
                "temperature": room_model.temperature,
            })

        logger.info("Rooms listed", count=len(rooms))
        return rooms

    except Exception as e:
        logger.error("Failed to list rooms", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/facilities")
async def list_facilities(
    db: AsyncSession = Depends(get_db),
):
    """List all facilities."""
    try:
        stmt = select(FacilityModel)
        result = await db.execute(stmt)
        facilities = result.scalars().all()

        return [
            {
                "id": str(f.id),
                "name": f.name,
                "code": f.code,
                "type": f.facility_type,
                "total_rooms": f.total_rooms,
                "is_operational": f.is_operational,
                "current_temperature": f.current_temperature,
                "current_energy_usage": f.current_energy_usage,
            }
            for f in facilities
        ]

    except Exception as e:
        logger.error("Failed to list facilities", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{room_id}")
async def get_room(
    room_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get room details by ID."""
    try:
        stmt = (
            select(RoomModel, FacilityModel)
            .join(FacilityModel, RoomModel.facility_id == FacilityModel.id)
            .where(RoomModel.id == room_id)
        )

        result = await db.execute(stmt)
        row = result.first()

        if not row:
            raise HTTPException(status_code=404, detail="Room not found")

        room_model, facility_model = row

        return {
            "id": str(room_model.id),
            "facility_id": str(facility_model.id),
            "facility_name": facility_model.name,
            "room_number": room_model.room_number,
            "room_type": room_model.room_type,
            "floor": room_model.floor,
            "capacity": room_model.capacity,
            "current_occupancy": room_model.current_occupancy,
            "is_available": room_model.is_available,
            "has_projector": room_model.has_projector,
            "has_whiteboard": room_model.has_whiteboard,
            "has_computers": room_model.has_computers,
            "has_wifi": True,
            "has_video_conference": room_model.has_video_conference,
            "temperature": room_model.temperature,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get room", error=str(e), room_id=str(room_id))
        raise HTTPException(status_code=500, detail=str(e))


# Request/Response Models for CRUD
class CreateFacilityRequest(BaseModel):
    """Create facility request."""
    name: str = Field(..., min_length=1, max_length=200)
    code: str = Field(..., min_length=1, max_length=20)
    facility_type: str = Field(..., max_length=50)
    total_rooms: int = Field(default=0, ge=0)


class UpdateFacilityRequest(BaseModel):
    """Update facility request."""
    name: str | None = Field(None, min_length=1, max_length=200)
    code: str | None = Field(None, min_length=1, max_length=20)
    facility_type: str | None = Field(None, max_length=50)
    total_rooms: int | None = Field(None, ge=0)
    is_operational: bool | None = None


class CreateRoomRequest(BaseModel):
    """Create room request."""
    facility_id: UUID
    room_number: str = Field(..., min_length=1, max_length=50)
    room_type: str = Field(..., max_length=50)
    building: str = Field(..., max_length=100)
    floor: int = Field(..., ge=0)
    capacity: int = Field(..., ge=1)
    area_sqm: float = Field(..., ge=0)
    has_projector: bool = Field(default=False)
    has_whiteboard: bool = Field(default=True)
    has_computers: bool = Field(default=False)
    computer_count: int = Field(default=0, ge=0)
    has_video_conference: bool = Field(default=False)
    is_available: bool = Field(default=True)
    is_bookable: bool = Field(default=True)


class UpdateRoomRequest(BaseModel):
    """Update room request."""
    room_number: str | None = Field(None, min_length=1, max_length=50)
    room_type: str | None = Field(None, max_length=50)
    building: str | None = Field(None, max_length=100)
    floor: int | None = Field(None, ge=0)
    capacity: int | None = Field(None, ge=1)
    area_sqm: float | None = Field(None, ge=0)
    has_projector: bool | None = None
    has_whiteboard: bool | None = None
    has_computers: bool | None = None
    computer_count: int | None = Field(None, ge=0)
    has_video_conference: bool | None = None
    is_available: bool | None = None
    is_bookable: bool | None = None
    temperature: float | None = None


@router.post("/facilities", status_code=status.HTTP_201_CREATED)
async def create_facility(
    request: CreateFacilityRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a new facility."""
    try:
        # Check if code already exists
        existing = await db.execute(
            select(FacilityModel).where(FacilityModel.code == request.code.upper())
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Facility with code {request.code} already exists"
            )

        facility = FacilityModel(
            name=request.name,
            code=request.code.upper(),
            facility_type=request.facility_type,
            total_rooms=request.total_rooms,
        )

        db.add(facility)
        await db.commit()
        await db.refresh(facility)

        logger.info("Facility created", facility_id=str(facility.id), code=facility.code)

        return {
            "id": str(facility.id),
            "name": facility.name,
            "code": facility.code,
            "type": facility.facility_type,
            "total_rooms": facility.total_rooms,
            "is_operational": facility.is_operational,
            "current_temperature": facility.current_temperature,
            "current_energy_usage": facility.current_energy_usage,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create facility", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/facilities/{facility_id}")
async def update_facility(
    facility_id: UUID,
    request: UpdateFacilityRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update facility information."""
    try:
        result = await db.execute(select(FacilityModel).where(FacilityModel.id == facility_id))
        facility = result.scalar_one_or_none()

        if not facility:
            raise HTTPException(status_code=404, detail="Facility not found")

        if request.name is not None:
            facility.name = request.name
        if request.code is not None:
            # Check if code is already taken
            existing = await db.execute(
                select(FacilityModel).where(
                    FacilityModel.code == request.code.upper(),
                    FacilityModel.id != facility_id
                )
            )
            if existing.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Facility code already in use"
                )
            facility.code = request.code.upper()
        if request.facility_type is not None:
            facility.facility_type = request.facility_type
        if request.total_rooms is not None:
            facility.total_rooms = request.total_rooms
        if request.is_operational is not None:
            facility.is_operational = request.is_operational

        facility.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(facility)

        logger.info("Facility updated", facility_id=str(facility_id))

        return {
            "id": str(facility.id),
            "name": facility.name,
            "code": facility.code,
            "type": facility.facility_type,
            "total_rooms": facility.total_rooms,
            "is_operational": facility.is_operational,
            "current_temperature": facility.current_temperature,
            "current_energy_usage": facility.current_energy_usage,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update facility", error=str(e), facility_id=str(facility_id))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/facilities/{facility_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_facility(
    facility_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a facility."""
    try:
        result = await db.execute(select(FacilityModel).where(FacilityModel.id == facility_id))
        facility = result.scalar_one_or_none()

        if not facility:
            raise HTTPException(status_code=404, detail="Facility not found")

        await db.delete(facility)
        await db.commit()

        logger.info("Facility deleted", facility_id=str(facility_id))

        return

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete facility", error=str(e), facility_id=str(facility_id))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_room(
    request: CreateRoomRequest,
    db: AsyncSession = Depends(get_db),
):
    """Create a new room."""
    try:
        # Verify facility exists
        facility_result = await db.execute(
            select(FacilityModel).where(FacilityModel.id == request.facility_id)
        )
        facility = facility_result.scalar_one_or_none()
        if not facility:
            raise HTTPException(status_code=404, detail="Facility not found")

        # Check if room number already exists in facility
        existing = await db.execute(
            select(RoomModel).where(
                RoomModel.facility_id == request.facility_id,
                RoomModel.room_number == request.room_number
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Room {request.room_number} already exists in this facility"
            )

        room = RoomModel(
            facility_id=request.facility_id,
            room_number=request.room_number,
            room_type=request.room_type,
            building=request.building,
            floor=request.floor,
            capacity=request.capacity,
            area_sqm=request.area_sqm,
            has_projector=request.has_projector,
            has_whiteboard=request.has_whiteboard,
            has_computers=request.has_computers,
            computer_count=request.computer_count,
            has_video_conference=request.has_video_conference,
            is_available=request.is_available,
            is_bookable=request.is_bookable,
        )

        db.add(room)
        await db.commit()
        await db.refresh(room)

        logger.info("Room created", room_id=str(room.id), room_number=room.room_number)

        return {
            "id": str(room.id),
            "facility_id": str(room.facility_id),
            "facility_name": facility.name,
            "room_number": room.room_number,
            "room_type": room.room_type,
            "floor": room.floor,
            "capacity": room.capacity,
            "current_occupancy": room.current_occupancy,
            "is_available": room.is_available,
            "has_projector": room.has_projector,
            "has_whiteboard": room.has_whiteboard,
            "has_computers": room.has_computers,
            "has_wifi": True,
            "has_video_conference": room.has_video_conference,
            "temperature": room.temperature,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create room", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{room_id}")
async def update_room(
    room_id: UUID,
    request: UpdateRoomRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update room information."""
    try:
        stmt = (
            select(RoomModel, FacilityModel)
            .join(FacilityModel, RoomModel.facility_id == FacilityModel.id)
            .where(RoomModel.id == room_id)
        )
        result = await db.execute(stmt)
        row = result.first()

        if not row:
            raise HTTPException(status_code=404, detail="Room not found")

        room, facility = row

        # Update fields
        if request.room_number is not None:
            # Check if room number is already taken in same facility
            existing = await db.execute(
                select(RoomModel).where(
                    RoomModel.facility_id == room.facility_id,
                    RoomModel.room_number == request.room_number,
                    RoomModel.id != room_id
                )
            )
            if existing.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Room number already exists in this facility"
                )
            room.room_number = request.room_number
        if request.room_type is not None:
            room.room_type = request.room_type
        if request.building is not None:
            room.building = request.building
        if request.floor is not None:
            room.floor = request.floor
        if request.capacity is not None:
            room.capacity = request.capacity
        if request.area_sqm is not None:
            room.area_sqm = request.area_sqm
        if request.has_projector is not None:
            room.has_projector = request.has_projector
        if request.has_whiteboard is not None:
            room.has_whiteboard = request.has_whiteboard
        if request.has_computers is not None:
            room.has_computers = request.has_computers
        if request.computer_count is not None:
            room.computer_count = request.computer_count
        if request.has_video_conference is not None:
            room.has_video_conference = request.has_video_conference
        if request.is_available is not None:
            room.is_available = request.is_available
        if request.is_bookable is not None:
            room.is_bookable = request.is_bookable
        if request.temperature is not None:
            room.temperature = request.temperature

        room.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(room)

        logger.info("Room updated", room_id=str(room_id))

        return {
            "id": str(room.id),
            "facility_id": str(room.facility_id),
            "facility_name": facility.name,
            "room_number": room.room_number,
            "room_type": room.room_type,
            "floor": room.floor,
            "capacity": room.capacity,
            "current_occupancy": room.current_occupancy,
            "is_available": room.is_available,
            "has_projector": room.has_projector,
            "has_whiteboard": room.has_whiteboard,
            "has_computers": room.has_computers,
            "has_wifi": True,
            "has_video_conference": room.has_video_conference,
            "temperature": room.temperature,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update room", error=str(e), room_id=str(room_id))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{room_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_room(
    room_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a room."""
    try:
        result = await db.execute(select(RoomModel).where(RoomModel.id == room_id))
        room = result.scalar_one_or_none()

        if not room:
            raise HTTPException(status_code=404, detail="Room not found")

        await db.delete(room)
        await db.commit()

        logger.info("Room deleted", room_id=str(room_id))

        return

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete room", error=str(e), room_id=str(room_id))
        raise HTTPException(status_code=500, detail=str(e))

