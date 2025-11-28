"""
GDPR Data Erasure API Endpoints

Provides endpoints for GDPR-compliant data erasure and pseudonymization.
"""

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from services.security_service.gdpr_erasure import GDPRDataErasureService
from shared.database import get_db

router = APIRouter(prefix="/gdpr", tags=["gdpr"])
logger = structlog.get_logger(__name__)


class DataErasureRequest(BaseModel):
    """GDPR data erasure request."""

    data_subject_id: UUID = Field(..., description="Student/user ID to erase")
    scope: str = Field(
        default="pseudonymize",
        description="'delete' (full deletion) or 'pseudonymize' (anonymize, preserve analytics)",
    )
    reason: str = Field(
        default="GDPR Right to be Forgotten",
        description="Reason for data erasure",
    )


class DataErasureResponse(BaseModel):
    """GDPR data erasure response."""

    status: str
    student_id: str
    records_affected: int
    analytics_preserved: bool
    pseudonym_id: str | None = None


async def get_current_user_id(authorization: str | None = Header(None)) -> UUID | None:
    """Extract user ID from JWT token."""
    if not authorization:
        return None

    try:
        from jose import jwt

        from shared.config import settings

        token = authorization.replace("Bearer ", "").strip()
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        return UUID(payload.get("sub"))

    except Exception as e:
        logger.warning("JWT validation failed", error=str(e))
        return None


async def verify_admin_or_self(
    authorization: str | None = Header(None),
    data_subject_id: UUID | None = None,
    db: AsyncSession = Depends(get_db),
) -> UUID:
    """
    Verify user is admin or requesting their own data erasure.

    Args:
        authorization: JWT token
        data_subject_id: Data subject ID
        db: Database session

    Returns:
        User ID of requester
    """
    user_id = await get_current_user_id(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    # Check if user is admin
    try:
        from jose import jwt

        from shared.config import settings

        token = authorization.replace("Bearer ", "").strip()
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        user_type = payload.get("user_type")

        # Admins can request erasure for anyone
        if user_type == "admin":
            return user_id

        # Users can only request erasure for themselves
        if data_subject_id and user_id == data_subject_id:
            return user_id

        raise HTTPException(
            status_code=403,
            detail="Access denied - Admin or self-request required"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Authorization verification failed", error=str(e))
        raise HTTPException(status_code=401, detail="Invalid authentication token")


@router.post("/erasure", response_model=DataErasureResponse, status_code=status.HTTP_200_OK)
async def request_data_erasure(
    request: DataErasureRequest,
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
) -> DataErasureResponse:
    """
    Request GDPR-compliant data erasure or pseudonymization.

    Supports:
    - Full deletion: Removes all personal data (analytics preserved as aggregated)
    - Pseudonymization: Replaces PII with anonymized identifiers (preserves analytics)

    Args:
        request: Data erasure request
        authorization: JWT token
        db: Database session

    Returns:
        DataErasureResponse: Erasure results
    """
    # Verify authorization
    requester_id = await verify_admin_or_self(
        authorization, request.data_subject_id, db
    )

    # Initialize GDPR service
    gdpr_service = GDPRDataErasureService(db)

    # Process erasure request
    result = await gdpr_service.request_erasure(
        data_subject_id=request.data_subject_id,
        requested_by=requester_id,
        scope=request.scope,
        reason=request.reason,
    )

    logger.info(
        "GDPR erasure completed",
        data_subject_id=str(request.data_subject_id),
        scope=request.scope,
        records_affected=result.get("records_affected", 0),
    )

    return DataErasureResponse(
        status=result["status"],
        student_id=result["student_id"],
        records_affected=result["records_affected"],
        analytics_preserved=result["analytics_preserved"],
        pseudonym_id=result.get("pseudonym_id"),
    )


@router.get("/verify/{student_id}", response_model=dict)
async def verify_erasure(
    student_id: UUID,
    authorization: str | None = Header(None),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Verify that data erasure was completed successfully.

    Args:
        student_id: Student ID to verify
        authorization: JWT token
        db: Database session

    Returns:
        Dictionary with verification results
    """
    # Verify authorization
    await verify_admin_or_self(authorization, student_id, db)

    # Initialize GDPR service
    gdpr_service = GDPRDataErasureService(db)

    # Verify erasure
    return await gdpr_service.verify_erasure(student_id)


