"""
User Service Dependencies

FastAPI dependencies for authentication and authorization.
"""


import structlog
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from services.user_service.auth_utils import TokenError, jwt_manager
from services.user_service.models import UserModel
from services.user_service.repository import UserRepository
from shared.database import get_db

logger = structlog.get_logger(__name__)

# Bearer token scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> UserModel:
    """
    Dependency to get current authenticated user from JWT token.

    Args:
        credentials: HTTP bearer credentials
        db: Database session

    Returns:
        UserModel: Authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials

    try:
        # Decode and validate token
        user_id = jwt_manager.get_user_id_from_token(token)

        # Get user from database
        user_repo = UserRepository(db)
        user = await user_repo.get_user_by_id(user_id)

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is inactive",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return user

    except TokenError as e:
        logger.warning("Token validation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(
    current_user: UserModel = Depends(get_current_user),
) -> UserModel:
    """
    Dependency to ensure user is active.

    Args:
        current_user: Current user from token

    Returns:
        UserModel: Active user

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user


async def require_admin(current_user: UserModel = Depends(get_current_user)) -> UserModel:
    """
    Dependency to require admin privileges.

    Args:
        current_user: Current user

    Returns:
        UserModel: Admin user

    Raises:
        HTTPException: If user is not an admin
    """
    if current_user.user_type != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user


async def require_student(current_user: UserModel = Depends(get_current_user)) -> UserModel:
    """Dependency to require student user type."""
    if current_user.user_type != "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Student account required",
        )
    return current_user


async def require_lecturer(current_user: UserModel = Depends(get_current_user)) -> UserModel:
    """Dependency to require lecturer user type."""
    if current_user.user_type != "lecturer":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Lecturer account required",
        )
    return current_user

