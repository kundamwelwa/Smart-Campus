"""
Authentication API Endpoints

Provides authentication, registration, and token management.
"""

from datetime import date, datetime
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from services.user_service.auth_utils import jwt_manager
from services.user_service.models import UserType
from services.user_service.repository import StudentRepository, UserRepository
from shared.database import get_db

logger = structlog.get_logger(__name__)

router = APIRouter()


# Request/Response Models
class RegisterRequest(BaseModel):
    """User registration request."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    middle_name: str | None = Field(default=None, max_length=100)
    user_type: str = Field(..., description="student, lecturer, staff, admin, guest")

    # Student-specific fields
    student_id: str | None = Field(default=None)
    major: str | None = Field(default=None)

    # Lecturer-specific fields
    employee_id: str | None = Field(default=None)
    department: str | None = Field(default=None)


class LoginRequest(BaseModel):
    """Login request."""

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Authentication token response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: UUID


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


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest, db: AsyncSession = Depends(get_db)
) -> TokenResponse:
    """
    Register a new user.

    Creates user account and issues authentication tokens.

    Args:
        request: Registration request
        db: Database session

    Returns:
        TokenResponse: Authentication tokens

    Raises:
        HTTPException: If registration fails
    """
    logger.info("Registration attempt", email=request.email, user_type=request.user_type)

    user_repo = UserRepository(db)

    # Check if user already exists
    existing_user = await user_repo.get_user_by_email(request.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists",
        )

    # Validate user_type
    try:
        user_type_enum = UserType(request.user_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid user_type. Must be one of: {', '.join([t.value for t in UserType])}",
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

    # Create type-specific record
    if user_type_enum == UserType.STUDENT:
        if not request.student_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Student ID is required for student registration",
            )

        student_repo = StudentRepository(db)
        await student_repo.create_student(
            user_id=user.id,
            student_id=request.student_id,
            enrollment_date=date.today(),
            major=request.major,
        )

    # Commit transaction
    await db.commit()

    # Generate tokens with role-based expiration
    access_token = jwt_manager.create_access_token(
        user_id=user.id,
        email=user.email,
        roles=user.role_ids or [],
        user_type=user.user_type.value if hasattr(user.user_type, 'value') else str(user.user_type),
    )

    refresh_token = jwt_manager.create_refresh_token(user_id=user.id)

    logger.info("User registered successfully", user_id=str(user.id), email=user.email)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=1800,  # 30 minutes
        user_id=user.id,
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)) -> TokenResponse:
    """
    Authenticate user and issue tokens.

    Args:
        request: Login request
        db: Database session

    Returns:
        TokenResponse: Authentication tokens

    Raises:
        HTTPException: If authentication fails
    """
    logger.info("Login attempt", email=request.email)

    user_repo = UserRepository(db)

    # Verify credentials
    user = await user_repo.verify_password(request.email, request.password)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    await db.commit()

    # Generate tokens with role-based expiration
    access_token = jwt_manager.create_access_token(
        user_id=user.id,
        email=user.email,
        roles=user.role_ids or [],
        user_type=user.user_type.value if hasattr(user.user_type, 'value') else str(user.user_type),
    )

    refresh_token = jwt_manager.create_refresh_token(user_id=user.id)

    logger.info("Login successful", user_id=str(user.id), email=user.email)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=1800,
        user_id=user.id,
    )


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(
    request: RefreshTokenRequest, db: AsyncSession = Depends(get_db)
) -> TokenResponse:
    """
    Refresh access token using refresh token.

    Args:
        refresh_token: Refresh token
        db: Database session

    Returns:
        TokenResponse: New tokens

    Raises:
        HTTPException: If refresh token is invalid
    """
    try:
        # Verify it's a refresh token
        if not jwt_manager.verify_token_type(request.refresh_token, "refresh"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type"
            )

        # Extract user ID
        user_id = jwt_manager.get_user_id_from_token(request.refresh_token)

        # Get user
        user_repo = UserRepository(db)
        user = await user_repo.get_user_by_id(user_id)

        if user is None or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive"
            )

        # Generate new tokens with role-based expiration
        new_access_token = jwt_manager.create_access_token(
            user_id=user.id,
            email=user.email,
            roles=user.role_ids or [],
            user_type=user.user_type.value if hasattr(user.user_type, 'value') else str(user.user_type),
        )

        new_refresh_token = jwt_manager.create_refresh_token(user_id=user.id)

        logger.info("Token refreshed", user_id=str(user.id))

        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=1800,
            user_id=user.id,
        )

    except Exception as e:
        logger.error("Token refresh failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

