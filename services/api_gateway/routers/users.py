"""
Users Router

Handles user profile and management operations.
Proxies requests to User Service.
"""

from datetime import datetime
from uuid import UUID

import httpx
import structlog
from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

router = APIRouter()

# User Service URL
USER_SERVICE_URL = "http://localhost:8001/api/v1"


# Response Models
class UserResponse(BaseModel):
    """User information response."""

    id: UUID
    email: str
    first_name: str
    last_name: str
    full_name: str
    user_type: str
    roles: list[str]
    is_active: bool
    created_at: datetime


@router.get("/me", response_model=UserResponse)
async def get_current_user(authorization: str | None = Header(None)) -> UserResponse:
    """
    Get current authenticated user's profile.

    Proxies to User Service.
    """
    logger.info("Get current user profile")

    try:
        headers = {}
        if authorization:
            headers["Authorization"] = authorization
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header required"
            )

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{USER_SERVICE_URL}/users/me",
                headers=headers,
                timeout=30.0
            )

            if response.status_code == 200:
                return UserResponse(**response.json())
            try:
                error_data = response.json()
                detail = error_data.get("detail", "Failed to fetch user profile")
            except Exception:
                detail = f"Failed to fetch user profile: {response.text[:200]}"

            logger.error("Get current user failed", status=response.status_code, error=detail)
            raise HTTPException(status_code=response.status_code, detail=detail)

    except httpx.RequestError as e:
        logger.error("Failed to connect to User Service", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User service unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error fetching user profile", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch user profile: {str(e)}"
        )

