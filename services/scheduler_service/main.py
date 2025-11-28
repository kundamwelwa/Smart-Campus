"""
Scheduler Service - Timetabling with Constraint Satisfaction

Uses OR-Tools CP-SAT solver for automatic timetable generation with soft/hard constraints.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from services.scheduler_service.constraint_solver import TimetableSolver
from shared.config import settings

logger = structlog.get_logger(__name__)

# Global solver instance
solver: TimetableSolver | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan."""
    global solver

    logger.info("Starting Scheduler Service")

    # Initialize constraint solver
    solver = TimetableSolver()
    logger.info("Timetable solver initialized")

    yield

    logger.info("Scheduler Service shutdown complete")


app = FastAPI(
    title="Argos Scheduler Service",
    description="Automated timetabling with constraint satisfaction",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
)


class Constraint(BaseModel):
    """Scheduling constraint."""

    type: str = Field(..., description="Constraint type: hard or soft")
    name: str
    weight: int = Field(default=1, description="Weight for soft constraints")


class ScheduleRequest(BaseModel):
    """Request for timetable generation."""

    sections: list[dict[str, Any]]
    rooms: list[dict[str, Any]]
    instructors: list[dict[str, Any]]
    constraints: list[Constraint] = Field(default_factory=list)


class ScheduleResponse(BaseModel):
    """Response with generated timetable."""

    success: bool
    assignments: dict[str, dict[str, Any]]
    conflicts: list[str]
    constraint_violations: list[str]
    optimization_score: float


@app.post("/api/v1/generate", response_model=ScheduleResponse)
async def generate_timetable(request: ScheduleRequest) -> ScheduleResponse:
    """
    Generate optimized timetable using constraint satisfaction.

    Considers:
    - Room capacity constraints (hard)
    - Instructor availability (hard)
    - Time slot conflicts (hard)
    - Room preferences (soft)
    - Balanced workload (soft)

    Args:
        request: Scheduling request

    Returns:
        Generated timetable with assignments
    """
    if solver is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Solver not initialized",
        )

    logger.info(
        "Timetable generation request",
        num_sections=len(request.sections),
        num_rooms=len(request.rooms),
    )

    # Solve scheduling problem
    result = await solver.solve(
        sections=request.sections,
        rooms=request.rooms,
        instructors=request.instructors,
        constraints=request.constraints,
    )

    logger.info(
        "Timetable generated",
        success=result['success'],
        assignments=len(result.get('assignments', {})),
    )

    return ScheduleResponse(
        success=result['success'],
        assignments=result.get('assignments', {}),
        conflicts=result.get('conflicts', []),
        constraint_violations=result.get('violations', []),
        optimization_score=result.get('score', 0.0),
    )


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check."""
    return {
        "status": "healthy",
        "service": "scheduler_service",
        "solver": "OR-Tools CP-SAT",
    }


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "service": "Scheduler Service",
        "version": "0.1.0",
        "solver": "Google OR-Tools",
        "features": ["constraint_satisfaction", "timetable_optimization"],
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.scheduler_service.main:app",
        host="0.0.0.0",
        port=settings.scheduler_service_port,
        reload=settings.debug,
    )

