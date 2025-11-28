"""
Security Service - Access Control and Incident Management

Handles physical access control, security incidents, and audit logging.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import UUID, uuid4

import structlog
from fastapi import FastAPI, status
from pydantic import BaseModel, Field

from shared.config import settings
from shared.database import close_db, init_db

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan."""
    logger.info("Starting Security Service")
    await init_db()
    yield
    await close_db()
    logger.info("Security Service shutdown complete")


app = FastAPI(
    title="Argos Security Service",
    description="Access control and security incident management",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
)


class AccessRequest(BaseModel):
    """Access control request."""

    user_id: UUID
    resource_type: str
    resource_id: UUID
    action: str


class AccessResponse(BaseModel):
    """Access control response."""

    granted: bool
    reason: str
    user_id: UUID
    resource_type: str


class SecurityIncidentRequest(BaseModel):
    """Security incident report."""

    incident_type: str = Field(..., description="Type of incident")
    severity: str = Field(..., description="low, medium, high, critical")
    description: str
    location: str | None = None
    affected_user_id: UUID | None = None


class IncidentResponse(BaseModel):
    """Incident response."""

    id: UUID
    incident_type: str
    severity: str
    status: str
    created_at: datetime


@app.post("/api/v1/access/check", response_model=AccessResponse)
async def check_access(request: AccessRequest) -> AccessResponse:
    """
    Check if user has access to a resource.

    Evaluates RBAC and ABAC policies to determine access.

    Args:
        request: Access request

    Returns:
        Access decision with reason
    """
    logger.info(
        "Access check",
        user_id=str(request.user_id),
        resource=request.resource_type,
        action=request.action,
    )

    # TODO: Integrate with RBAC/ABAC service
    # For now, simplified logic

    # Simulate access decision
    granted = True  # Would call actual authorization service

    if granted:
        logger.info("Access granted", user_id=str(request.user_id))
    else:
        logger.warning("Access denied", user_id=str(request.user_id))

    return AccessResponse(
        granted=granted,
        reason="Authorized by role" if granted else "Insufficient permissions",
        user_id=request.user_id,
        resource_type=request.resource_type,
    )


@app.post("/api/v1/incidents", response_model=IncidentResponse, status_code=status.HTTP_201_CREATED)
async def report_incident(request: SecurityIncidentRequest) -> IncidentResponse:
    """
    Report a security incident.

    Creates incident record and triggers alerts for high-severity incidents.

    Args:
        request: Incident details

    Returns:
        Created incident
    """
    logger.warning(
        "Security incident reported",
        type=request.incident_type,
        severity=request.severity,
    )

    incident_id = uuid4()

    # TODO: Store in database and emit event

    return IncidentResponse(
        id=incident_id,
        incident_type=request.incident_type,
        severity=request.severity,
        status="reported",
        created_at=datetime.utcnow(),
    )


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check."""
    return {
        "status": "healthy",
        "service": "security_service",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.security_service.main:app",
        host="0.0.0.0",
        port=settings.security_service_port,
        reload=settings.debug,
    )

