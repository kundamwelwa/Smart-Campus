"""
Facilities endpoint proxies.
"""

from uuid import UUID

import httpx
import structlog
from fastapi import APIRouter, Body, Header, HTTPException, Request, status

from shared.config import settings

router = APIRouter(prefix="/facilities", tags=["facilities"])
logger = structlog.get_logger(__name__)


@router.get("/rooms")
async def list_rooms(
    facility_id: str | None = None,
    is_available: bool | None = None,
    min_capacity: int | None = None,
    authorization: str | None = Header(None),
):
    """Proxy to Facility Service for rooms list."""
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
                f"http://localhost:{settings.facility_service_port}/api/v1/facilities/rooms",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            return response.json()

    except httpx.RequestError as e:
        logger.error("Failed to proxy facilities rooms", error=str(e))
        raise HTTPException(status_code=503, detail="Facility Service unavailable")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("Unexpected error proxying facilities rooms", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rooms/facilities")
async def list_facilities(
    authorization: str | None = Header(None),
):
    """Proxy to Facility Service for facilities list."""
    try:
        headers = {}
        if authorization:
            headers["Authorization"] = authorization

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"http://localhost:{settings.facility_service_port}/api/v1/facilities/rooms/facilities",
                headers=headers,
            )
            if response.status_code == 200:
                return response.json()
            try:
                error_data = response.json()
                detail = error_data.get("detail", f"Failed to fetch facilities: {response.status_code}")
            except Exception:
                detail = f"Failed to fetch facilities: {response.text[:200]}"
            raise HTTPException(status_code=response.status_code, detail=detail)

    except httpx.RequestError as e:
        logger.error("Failed to proxy facilities list", error=str(e))
        raise HTTPException(status_code=503, detail="Facility Service unavailable")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("Unexpected error proxying facilities list", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rooms/facilities", status_code=status.HTTP_201_CREATED)
async def create_facility(
    request: Request,
    authorization: str | None = Header(None),
):
    """Proxy to Facility Service for creating facility."""
    try:
        headers = {"Content-Type": "application/json"}
        if authorization:
            headers["Authorization"] = authorization

        json_data = await request.json()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"http://localhost:{settings.facility_service_port}/api/v1/facilities/rooms/facilities",
                headers=headers,
                json=json_data,
            )
            response.raise_for_status()
            return response.json()

    except httpx.RequestError as e:
        logger.error("Failed to proxy facility creation", error=str(e))
        raise HTTPException(status_code=503, detail="Facility Service unavailable")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("Unexpected error proxying facility creation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/rooms/facilities/{facility_id}")
async def update_facility(
    facility_id: UUID,
    request: Request,
    authorization: str | None = Header(None),
):
    """Proxy to Facility Service for updating facility."""
    try:
        headers = {"Content-Type": "application/json"}
        if authorization:
            headers["Authorization"] = authorization

        json_data = await request.json()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.put(
                f"http://localhost:{settings.facility_service_port}/api/v1/facilities/rooms/facilities/{facility_id}",
                headers=headers,
                json=json_data,
            )
            response.raise_for_status()
            return response.json()

    except httpx.RequestError as e:
        logger.error("Failed to proxy facility update", error=str(e))
        raise HTTPException(status_code=503, detail="Facility Service unavailable")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("Unexpected error proxying facility update", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rooms/facilities/{facility_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_facility(
    facility_id: UUID,
    authorization: str | None = Header(None),
):
    """Proxy to Facility Service for deleting facility."""
    try:
        headers = {}
        if authorization:
            headers["Authorization"] = authorization

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"http://localhost:{settings.facility_service_port}/api/v1/facilities/rooms/facilities/{facility_id}",
                headers=headers,
            )
            response.raise_for_status()
            return

    except httpx.RequestError as e:
        logger.error("Failed to proxy facility deletion", error=str(e))
        raise HTTPException(status_code=503, detail="Facility Service unavailable")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("Unexpected error proxying facility deletion", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rooms", status_code=status.HTTP_201_CREATED)
async def create_room(
    request: Request,
    authorization: str | None = Header(None),
):
    """Proxy to Facility Service for creating room."""
    try:
        headers = {"Content-Type": "application/json"}
        if authorization:
            headers["Authorization"] = authorization

        json_data = await request.json()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"http://localhost:{settings.facility_service_port}/api/v1/facilities/rooms",
                headers=headers,
                json=json_data,
            )
            response.raise_for_status()
            return response.json()

    except httpx.RequestError as e:
        logger.error("Failed to proxy room creation", error=str(e))
        raise HTTPException(status_code=503, detail="Facility Service unavailable")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("Unexpected error proxying room creation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/rooms/{room_id}")
async def update_room(
    room_id: UUID,
    request: Request,
    authorization: str | None = Header(None),
):
    """Proxy to Facility Service for updating room."""
    try:
        headers = {"Content-Type": "application/json"}
        if authorization:
            headers["Authorization"] = authorization

        json_data = await request.json()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.put(
                f"http://localhost:{settings.facility_service_port}/api/v1/facilities/rooms/{room_id}",
                headers=headers,
                json=json_data,
            )
            response.raise_for_status()
            return response.json()

    except httpx.RequestError as e:
        logger.error("Failed to proxy room update", error=str(e))
        raise HTTPException(status_code=503, detail="Facility Service unavailable")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("Unexpected error proxying room update", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/rooms/{room_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_room(
    room_id: UUID,
    authorization: str | None = Header(None),
):
    """Proxy to Facility Service for deleting room."""
    try:
        headers = {}
        if authorization:
            headers["Authorization"] = authorization

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.delete(
                f"http://localhost:{settings.facility_service_port}/api/v1/facilities/rooms/{room_id}",
                headers=headers,
            )
            response.raise_for_status()
            return

    except httpx.RequestError as e:
        logger.error("Failed to proxy room deletion", error=str(e))
        raise HTTPException(status_code=503, detail="Facility Service unavailable")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("Unexpected error proxying room deletion", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bookings")
async def create_booking(
    booking_data: dict = Body(...),
    authorization: str | None = Header(None),
):
    """Proxy to Facility Service for creating room bookings."""
    try:
        headers = {"Content-Type": "application/json"}
        if authorization:
            headers["Authorization"] = authorization

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"http://localhost:{settings.facility_service_port}/api/v1/facilities/bookings",
                headers=headers,
                json=booking_data,
            )
            response.raise_for_status()
            return response.json()

    except httpx.RequestError as e:
        logger.error("Failed to proxy booking creation", error=str(e))
        raise HTTPException(status_code=503, detail="Facility Service unavailable")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("Unexpected error proxying booking creation", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

