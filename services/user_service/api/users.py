"""
User Management API Endpoints

CRUD operations for user management.
"""

from datetime import datetime
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from services.user_service.dependencies import get_current_user
from services.user_service.repository import UserRepository
from shared.database import get_db

logger = structlog.get_logger(__name__)

router = APIRouter()


class UserProfileResponse(BaseModel):
    """User profile response."""

    id: UUID
    email: str
    first_name: str
    last_name: str
    full_name: str
    user_type: str
    roles: list[str]
    attached_roles: list[str]
    is_active: bool
    email_verified: bool
    created_at: datetime
    last_login_at: datetime | None


@router.get("/me", response_model=UserProfileResponse)
async def get_current_user_profile(
    current_user=Depends(get_current_user),
) -> UserProfileResponse:
    """
    Get current authenticated user's profile.

    Returns:
        UserProfileResponse: User profile
    """
    return UserProfileResponse(
        id=current_user.id,
        email=current_user.email,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        full_name=f"{current_user.first_name} {current_user.last_name}",
        user_type=current_user.user_type.value,
        roles=current_user.role_ids or [],
        attached_roles=current_user.attached_roles or [],
        is_active=current_user.is_active,
        email_verified=current_user.email_verified,
        created_at=current_user.created_at,
        last_login_at=current_user.last_login_at,
    )


@router.get("/{user_id}", response_model=UserProfileResponse)
async def get_user_by_id(
    user_id: UUID, db: AsyncSession = Depends(get_db)
) -> UserProfileResponse:
    """
    Get user by ID (admin only).

    Args:
        user_id: User UUID
        db: Database session

    Returns:
        UserProfileResponse: User profile

    Raises:
        HTTPException: If user not found
    """
    user_repo = UserRepository(db)
    user = await user_repo.get_user_by_id(user_id)

    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return UserProfileResponse(
        id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        full_name=f"{user.first_name} {user.last_name}",
        user_type=user.user_type.value,
        roles=user.role_ids or [],
        attached_roles=user.attached_roles or [],
        is_active=user.is_active,
        email_verified=user.email_verified,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
    )

