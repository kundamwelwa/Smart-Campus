"""
Lecturer-specific endpoint proxies.
"""


import httpx
import structlog
from fastapi import APIRouter, Header, HTTPException

from shared.config import settings

router = APIRouter(prefix="/lecturer", tags=["lecturer"])
logger = structlog.get_logger(__name__)


@router.get("/sections")
async def get_lecturer_sections(
    authorization: str | None = Header(None),
):
    """Proxy to Academic Service for lecturer sections."""
    try:
        headers = {}
        if authorization:
            headers["Authorization"] = authorization

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"http://localhost:{settings.academic_service_port}/api/v1/lecturer/sections",
                headers=headers,
            )
            response.raise_for_status()
            return response.json()

    except httpx.RequestError as e:
        logger.error("Failed to proxy lecturer sections", error=str(e))
        raise HTTPException(status_code=503, detail="Academic Service unavailable")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("Unexpected error proxying lecturer sections", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assessments")
async def get_lecturer_assessments(
    authorization: str | None = Header(None),
):
    """Proxy to Academic Service for lecturer assessments."""
    try:
        headers = {}
        if authorization:
            headers["Authorization"] = authorization

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"http://localhost:{settings.academic_service_port}/api/v1/lecturer/assessments",
                headers=headers,
            )
            response.raise_for_status()
            return response.json()

    except httpx.RequestError as e:
        logger.error("Failed to proxy lecturer assessments", error=str(e))
        raise HTTPException(status_code=503, detail="Academic Service unavailable")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("Unexpected error proxying lecturer assessments", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/students")
async def get_lecturer_students(
    section_id: str | None = None,
    authorization: str | None = Header(None),
):
    """Proxy to Academic Service for lecturer students."""
    try:
        headers = {}
        if authorization:
            headers["Authorization"] = authorization

        params = {}
        if section_id:
            params["section_id"] = section_id

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"http://localhost:{settings.academic_service_port}/api/v1/lecturer/students",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            return response.json()

    except httpx.RequestError as e:
        logger.error("Failed to proxy lecturer students", error=str(e))
        raise HTTPException(status_code=503, detail="Academic Service unavailable")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("Unexpected error proxying lecturer students", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

