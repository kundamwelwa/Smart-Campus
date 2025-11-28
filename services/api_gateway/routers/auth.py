"""
Authentication Router

Handles user authentication, registration, and token management.
Proxies requests to User Service.
"""

from datetime import datetime
from uuid import UUID

import httpx
import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr, Field

logger = structlog.get_logger(__name__)

router = APIRouter()

# User Service URL
USER_SERVICE_URL = "http://localhost:8001/api/v1"


# Request/Response Models
class RegisterRequest(BaseModel):
    """User registration request."""

    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    middle_name: str | None = Field(default=None, max_length=100)
    user_type: str = Field(..., description="student, lecturer, staff, admin")

    # Student-specific fields
    student_id: str | None = Field(default=None)
    major: str | None = Field(default=None)

    # Lecturer-specific fields
    employee_id: str | None = Field(default=None)
    department: str | None = Field(default=None)


class LoginRequest(BaseModel):
    """User login request."""

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
    user_type: str
    roles: list[str]
    created_at: datetime


class LogoutRequest(BaseModel):
    """Logout request payload."""

    access_token: str


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest) -> TokenResponse:
    """
    Register a new user.

    Proxies request to User Service.

    Args:
        request: Registration request

    Returns:
        TokenResponse: Authentication tokens

    Raises:
        HTTPException: If registration fails
    """
    logger.info("User registration attempt", email=request.email, user_type=request.user_type)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{USER_SERVICE_URL}/auth/register",
                json=request.model_dump(),
                timeout=30.0
            )

            if response.status_code == 201:
                logger.info("User registered successfully", email=request.email)
                return TokenResponse(**response.json())
            # Try to parse error from response
            try:
                error_data = response.json()
                error_detail = error_data.get("detail", "Registration failed")
            except Exception:
                error_detail = f"Registration failed: {response.text[:200]}"

            logger.error("Registration failed", email=request.email, status=response.status_code, error=error_detail)
            raise HTTPException(
                status_code=response.status_code,
                detail=error_detail
            )
    except httpx.RequestError as e:
        logger.error("Failed to connect to User Service", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User service unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error during registration", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest) -> TokenResponse:
    """
    Authenticate user and issue tokens.

    Proxies request to User Service.

    Args:
        request: Login request

    Returns:
        TokenResponse: Access and refresh tokens

    Raises:
        HTTPException: If authentication fails
    """
    logger.info("Login attempt", email=request.email)

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{USER_SERVICE_URL}/auth/login",
                json=request.model_dump(),
                timeout=30.0
            )

            if response.status_code == 200:
                logger.info("Login successful", email=request.email)
                return TokenResponse(**response.json())
            # Try to parse error from response
            try:
                error_data = response.json()
                error_detail = error_data.get("detail", "Authentication failed")
            except Exception:
                error_detail = f"Authentication failed: {response.text[:200]}"

            logger.error("Login failed", email=request.email, status=response.status_code, error=error_detail)
            raise HTTPException(
                status_code=response.status_code,
                detail=error_detail
            )
    except httpx.RequestError as e:
        logger.error("Failed to connect to User Service", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="User service unavailable"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Unexpected error during login", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )


class RefreshTokenRequest(BaseModel):
    refresh_token: str


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest) -> TokenResponse:
    """
    Refresh access token using refresh token.

    Proxies to User Service.
    """
    logger.info("Token refresh attempt")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{USER_SERVICE_URL}/auth/refresh",
                json={"refresh_token": request.refresh_token},
                timeout=30.0
            )

            if response.status_code == 200:
                logger.info("Token refreshed successfully")
                return TokenResponse(**response.json())
            try:
                error_data = response.json()
                detail = error_data.get("detail", "Token refresh failed")
            except Exception:
                detail = f"Token refresh failed: {response.text[:200]}"

            logger.error("Token refresh failed", status=response.status_code, error=detail)
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
        logger.error("Unexpected error during token refresh", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token refresh failed: {str(e)}"
        )


@router.post("/logout")
async def logout(payload: LogoutRequest) -> dict[str, str]:
    """
    Logout user and invalidate tokens.

    Returns:
        dict: Logout confirmation
    """
    logger.info("Logout attempt")

    # TODO: Implement token invalidation (blacklist in Redis)

    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def get_current_user() -> UserResponse:
    """
    Get current authenticated user information.

    Returns:
        UserResponse: Current user data

    Raises:
        HTTPException: If not authenticated
    """
    # TODO: Extract user from JWT token and fetch from User Service

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="User profile endpoint will be implemented",
    )

