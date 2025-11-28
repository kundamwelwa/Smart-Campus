"""
Admin-specific endpoints for user management.
"""

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from services.user_service.auth_utils import password_hasher
from services.user_service.models import UserModel, UserType
from services.user_service.repository import UserRepository
from shared.database import get_db

router = APIRouter(prefix="/admin", tags=["admin"])
logger = structlog.get_logger(__name__)


async def verify_admin(
    authorization: str | None = Header(None, alias="Authorization"),
    db: AsyncSession = Depends(get_db),
) -> UUID:
    """Verify user is admin and return admin ID."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    try:
        from jose import jwt
        from sqlalchemy import select

        from shared.config import settings

        # Extract token - handle both "Bearer token" and just "token"
        token = authorization.replace("Bearer ", "").strip()

        # Decode JWT
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        user_id = UUID(payload.get("sub"))

        # Fetch user from database to verify admin status
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        if not user.is_active:
            raise HTTPException(status_code=403, detail="User account is inactive")

        # Verify user is admin
        if user.user_type.value != "admin":
            raise HTTPException(status_code=403, detail="Access denied - Admin role required")

        return user_id

    except HTTPException:
        raise
    except Exception as e:
        logger.error("JWT validation failed", error=str(e))
        raise HTTPException(status_code=401, detail="Invalid authentication token")


@router.get("/users")
async def list_all_users(
    user_type: str | None = None,
    is_active: bool | None = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    admin_id: UUID = Depends(verify_admin),
):
    """
    List all users (admin only).

    Args:
        user_type: Filter by user type
        is_active: Filter by active status
        limit: Maximum number of users to return
        offset: Number of users to skip

    Returns:
        List of users
    """
    try:
        stmt = select(UserModel)

        if user_type:
            stmt = stmt.where(UserModel.user_type == user_type)
        if is_active is not None:
            stmt = stmt.where(UserModel.is_active == is_active)

        stmt = stmt.limit(limit).offset(offset)

        result = await db.execute(stmt)
        users = result.scalars().all()

        users_list = []
        for user in users:
            users_list.append({
                "id": str(user.id),
                "email": user.email,
                "first_name": user.first_name,
                "last_name": user.last_name,
                "full_name": f"{user.first_name} {user.last_name}",
                "user_type": user.user_type,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
            })

        logger.info(
            "Admin listed users",
            admin_id=str(admin_id),
            count=len(users_list),
        )

        return users_list

    except Exception as e:
        logger.error("Failed to list users", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/stats")
async def get_user_statistics(
    db: AsyncSession = Depends(get_db),
    admin_id: UUID = Depends(verify_admin),
):
    """
    Get user statistics (admin only).

    Returns:
        Dictionary with user counts by type and status
    """
    try:
        # Total users
        total_stmt = select(func.count(UserModel.id))
        total_result = await db.execute(total_stmt)
        total_users = total_result.scalar() or 0

        # By type
        type_stmt = select(UserModel.user_type, func.count(UserModel.id)).group_by(
            UserModel.user_type
        )
        type_result = await db.execute(type_stmt)
        by_type = {row[0]: row[1] for row in type_result.all()}

        # Active users
        active_stmt = select(func.count(UserModel.id)).where(UserModel.is_active)
        active_result = await db.execute(active_stmt)
        active_users = active_result.scalar() or 0

        return {
            "total_users": total_users,
            "active_users": active_users,
            "inactive_users": total_users - active_users,
            "by_type": by_type,
        }

    except Exception as e:
        logger.error("Failed to get user statistics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/users/{user_id}/activate")
async def activate_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    admin_id: UUID = Depends(verify_admin),
):
    """Activate a user account (admin only)."""
    try:
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user.is_active = True
        await db.commit()

        logger.info("User activated", user_id=str(user_id), admin_id=str(admin_id))

        return {"message": "User activated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to activate user", error=str(e), user_id=str(user_id))
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/users/{user_id}/deactivate")
async def deactivate_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    admin_id: UUID = Depends(verify_admin),
):
    """Deactivate a user account (admin only)."""
    try:
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Prevent deactivating yourself
        if user.id == admin_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot deactivate your own account"
            )

        user.is_active = False
        await db.commit()

        logger.info("User deactivated", user_id=str(user_id), admin_id=str(admin_id))

        return {"message": "User deactivated successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to deactivate user", error=str(e), user_id=str(user_id))
        raise HTTPException(status_code=500, detail=str(e))


# Request/Response Models for User CRUD
class CreateUserRequest(BaseModel):
    """Create user request (admin only)."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    middle_name: str | None = Field(default=None, max_length=100)
    user_type: str = Field(..., description="student, lecturer, staff, admin, guest")
    is_active: bool = Field(default=True)


class UpdateUserRequest(BaseModel):
    """Update user request (admin only)."""
    email: EmailStr | None = None
    first_name: str | None = Field(None, min_length=1, max_length=100)
    last_name: str | None = Field(None, min_length=1, max_length=100)
    middle_name: str | None = Field(None, max_length=100)
    user_type: str | None = None
    is_active: bool | None = None
    password: str | None = Field(None, min_length=8)


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(
    request: CreateUserRequest,
    db: AsyncSession = Depends(get_db),
    admin_id: UUID = Depends(verify_admin),
):
    """Create a new user (admin only)."""
    try:
        user_repo = UserRepository(db)

        # Check if user already exists
        existing_user = await user_repo.get_user_by_email(request.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )

        # Validate user_type
        try:
            UserType(request.user_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid user_type. Must be one of: {', '.join([t.value for t in UserType])}"
            )

        # Create user
        user = await user_repo.create_user(
            email=request.email,
            password=request.password,
            first_name=request.first_name,
            last_name=request.last_name,
            middle_name=request.middle_name,
            user_type=request.user_type,
            consent_given=True,
        )

        # Set active status
        user.is_active = request.is_active
        await db.commit()
        await db.refresh(user)

        logger.info("User created by admin", user_id=str(user.id), admin_id=str(admin_id))

        return {
            "id": str(user.id),
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "full_name": f"{user.first_name} {user.last_name}",
            "user_type": user.user_type.value,
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create user", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/users/{user_id}")
async def update_user(
    user_id: UUID,
    request: UpdateUserRequest,
    db: AsyncSession = Depends(get_db),
    admin_id: UUID = Depends(verify_admin),
):
    """Update user information (admin only)."""
    try:
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Update fields
        if request.email is not None:
            # Check if email is already taken by another user
            existing = await db.execute(
                select(UserModel).where(UserModel.email == request.email, UserModel.id != user_id)
            )
            if existing.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already in use by another user"
                )
            user.email = request.email

        if request.first_name is not None:
            user.first_name = request.first_name
        if request.last_name is not None:
            user.last_name = request.last_name
        if request.middle_name is not None:
            user.middle_name = request.middle_name
        if request.user_type is not None:
            try:
                user.user_type = UserType(request.user_type)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid user_type. Must be one of: {', '.join([t.value for t in UserType])}"
                )
        if request.is_active is not None:
            user.is_active = request.is_active
        if request.password is not None:
            user.password_hash = password_hasher.hash_password(request.password)

        await db.commit()
        await db.refresh(user)

        logger.info("User updated by admin", user_id=str(user_id), admin_id=str(admin_id))

        return {
            "id": str(user.id),
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "full_name": f"{user.first_name} {user.last_name}",
            "user_type": user.user_type.value,
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat(),
            "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update user", error=str(e), user_id=str(user_id))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: UUID,
    db: AsyncSession = Depends(get_db),
    admin_id: UUID = Depends(verify_admin),
):
    """Delete a user (admin only)."""
    try:
        stmt = select(UserModel).where(UserModel.id == user_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Prevent deleting yourself
        if user.id == admin_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )

        await db.delete(user)
        await db.commit()

        logger.info("User deleted by admin", user_id=str(user_id), admin_id=str(admin_id))

        return

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete user", error=str(e), user_id=str(user_id))
        raise HTTPException(status_code=500, detail=str(e))

