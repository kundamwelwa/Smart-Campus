"""Facility Router - Room booking and facility management"""

import httpx
import structlog
from fastapi import APIRouter, Header, HTTPException, Query

from shared.config import settings

router = APIRouter()
logger = structlog.get_logger(__name__)

FACILITY_SERVICE_URL = f"http://localhost:{settings.facility_service_port}/api/v1/facilities"


@router.get("/rooms")
async def list_rooms(
    facility_id: str | None = Query(None),
    is_available: bool | None = Query(None),
    min_capacity: int | None = Query(None),
    authorization: str | None = Header(None),
):
    """
    List available rooms.

    Proxies to Facility Service.
    """
    logger.info("List rooms", facility_id=facility_id, is_available=is_available)

    try:
        headers = {}
        if authorization:
            headers["Authorization"] = authorization

        params = {}
        if facility_id:
            params["facility_id"] = facility_id
        if is_available is not None:
            params["is_available"] = is_available
        if min_capacity:
            params["min_capacity"] = min_capacity

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{FACILITY_SERVICE_URL}/rooms",
                headers=headers,
                params=params,
            )

            if response.status_code == 200:
                return response.json()
            try:
                error_data = response.json()
                detail = error_data.get("detail", "Failed to fetch rooms")
            except Exception:
                detail = f"Failed to fetch rooms: {response.text[:200]}"

            raise HTTPException(status_code=response.status_code, detail=detail)

    except httpx.RequestError as e:
        logger.error("Failed to connect to Facility Service", error=str(e))
        raise HTTPException(
            status_code=503,
            detail="Facility service unavailable. Please ensure the facility service is running."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error fetching rooms", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

