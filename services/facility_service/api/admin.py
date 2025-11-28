"""
Admin-specific endpoints for Facility Service.
"""

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from services.facility_service.models import BookingModel, FacilityModel, RoomModel
from shared.database import get_db

router = APIRouter(prefix="/admin", tags=["admin"])
logger = structlog.get_logger(__name__)


async def verify_admin(authorization: str | None = Header(None)) -> UUID:
    """Verify user is admin and return admin ID."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    try:
        import httpx
        from jose import jwt

        from shared.config import settings

        # Extract token
        token = authorization.replace("Bearer ", "").strip()

        # Decode JWT
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        user_id = UUID(payload.get("sub"))

        # Verify admin status by calling user service - PRODUCTION: No fallbacks
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"http://localhost:{settings.user_service_port}/api/v1/users/me",
                headers={"Authorization": authorization},
            )
            response.raise_for_status()

            user_data = response.json()
            if user_data.get("user_type") != "admin":
                raise HTTPException(status_code=403, detail="Access denied - Admin role required")

        return user_id

    except HTTPException:
        raise
    except Exception as e:
        logger.error("JWT validation failed", error=str(e))
        raise HTTPException(status_code=401, detail="Invalid authentication token")


@router.get("/stats")
async def get_facility_statistics(
    db: AsyncSession = Depends(get_db),
    authorization: str | None = Header(None),
):
    """
    Get facility service statistics (admin only).

    Returns:
        Dictionary with facility statistics
    """
    # Optional auth - allow internal service calls
    try:
        # Total facilities
        total_facilities_stmt = select(func.count(FacilityModel.id))
        total_facilities_result = await db.execute(total_facilities_stmt)
        total_facilities = total_facilities_result.scalar() or 0

        # Total rooms
        total_rooms_stmt = select(func.count(RoomModel.id))
        total_rooms_result = await db.execute(total_rooms_stmt)
        total_rooms = total_rooms_result.scalar() or 0

        # Available rooms
        available_rooms_stmt = select(func.count(RoomModel.id)).where(
            RoomModel.is_available
        )
        available_rooms_result = await db.execute(available_rooms_stmt)
        available_rooms = available_rooms_result.scalar() or 0

        # Total bookings
        total_bookings_stmt = select(func.count(BookingModel.id))
        total_bookings_result = await db.execute(total_bookings_stmt)
        total_bookings = total_bookings_result.scalar() or 0

        # Active bookings (future bookings)
        from datetime import datetime
        active_bookings_stmt = select(func.count(BookingModel.id)).where(
            BookingModel.end_time >= datetime.utcnow()
        )
        active_bookings_result = await db.execute(active_bookings_stmt)
        active_bookings = active_bookings_result.scalar() or 0

        # Rooms by building
        building_stmt = select(RoomModel.building, func.count(RoomModel.id)).group_by(
            RoomModel.building
        )
        building_result = await db.execute(building_stmt)
        by_building = {row[0]: row[1] for row in building_result.all()}

        return {
            "total_facilities": total_facilities,
            "total_rooms": total_rooms,
            "available_rooms": available_rooms,
            "occupied_rooms": total_rooms - available_rooms,
            "total_bookings": total_bookings,
            "active_bookings": active_bookings,
            "by_building": by_building,
        }

    except Exception as e:
        logger.error("Failed to get facility statistics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

