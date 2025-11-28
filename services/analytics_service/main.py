"""
Analytics Service - ML Model Serving

Serves machine learning models for predictions and recommendations.
Provides explainable AI capabilities.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI
from pydantic import BaseModel, Field

from shared.config import settings

# Try to import ML models - they're optional if dependencies aren't installed
try:
    from ml.models.enrollment_predictor import EnrollmentPredictor
    from ml.models.room_optimizer import RoomUsageOptimizer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    EnrollmentPredictor = None
    RoomUsageOptimizer = None

logger = structlog.get_logger(__name__)

if not ML_AVAILABLE:
    logger.warning(
        "ML packages not available - Analytics service will return mock predictions",
        details="Install pytorch-lightning, stable-baselines3, gym to enable ML features"
    )

# Global model instances
enrollment_predictor: EnrollmentPredictor | None = None
room_optimizer: RoomUsageOptimizer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan - load ML models."""
    global enrollment_predictor, room_optimizer

    logger.info("Starting Analytics Service")

    if ML_AVAILABLE:
        logger.info("ML packages available - Loading models")
        # Load enrollment predictor
        try:
            enrollment_predictor = EnrollmentPredictor()
            model_path = Path(settings.ml_model_path) / 'enrollment_predictor.pt'

            if model_path.exists():
                await enrollment_predictor.load(model_path)
                logger.info("Enrollment predictor loaded", path=str(model_path))
            else:
                logger.info("Enrollment predictor model not found - using untrained model (this is normal for initial setup)")
        except Exception as e:
            logger.error("Failed to load enrollment predictor", error=str(e))
            enrollment_predictor = EnrollmentPredictor()  # Use untrained model

        # Initialize room optimizer
        room_optimizer = RoomUsageOptimizer()
        logger.info("Room optimizer initialized")
    else:
        logger.warning("ML packages not installed - Analytics will use fallback predictions")

    logger.info("Analytics Service ready")

    yield

    logger.info("Analytics Service shutdown complete")


app = FastAPI(
    title="Argos Analytics Service",
    description="ML model serving with explainable AI",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# Request/Response Models
class EnrollmentPredictionRequest(BaseModel):
    """Request for enrollment prediction."""

    student_id: str
    gpa: float = Field(..., ge=0.0, le=4.0)
    credits_enrolled: int = Field(..., ge=0, le=24)
    attendance_rate: float = Field(..., ge=0.0, le=100.0)
    engagement_score: float = Field(..., ge=0.0, le=1.0)
    previous_dropout_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    course_difficulty: float = Field(default=0.5, ge=0.0, le=1.0)
    study_hours: float = Field(default=10.0, ge=0.0, le=60.0)
    num_failed_courses: int = Field(default=0, ge=0)
    explain: bool = Field(default=True, description="Include explanation")


class EnrollmentPredictionResponse(BaseModel):
    """Response with dropout prediction."""

    student_id: str
    dropout_probability: float
    retention_probability: float
    risk_level: str
    confidence: float
    explanation: dict | None = None


class RoomOptimizationRequest(BaseModel):
    """Request for room optimization."""

    sections: list[dict]
    rooms: list[dict]
    explain: bool = Field(default=True)


class RoomOptimizationResponse(BaseModel):
    """Response with optimal room allocation."""

    allocation: dict[int, int]
    metrics: dict[str, float]
    num_sections_allocated: int
    num_violations: int
    explanation: dict | None = None


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check with model status."""
    return {
        "status": "healthy",
        "service": "analytics_service",
        "models_loaded": {
            "enrollment_predictor": enrollment_predictor is not None and enrollment_predictor.is_trained,
            "room_optimizer": room_optimizer is not None,
        },
    }


@app.post("/api/v1/predict/enrollment", response_model=EnrollmentPredictionResponse)
async def predict_enrollment(request: EnrollmentPredictionRequest) -> EnrollmentPredictionResponse:
    """
    Predict student dropout probability - REAL ML MODEL!

    Uses LSTM model with attention mechanism to predict dropout risk
    based on academic performance and engagement metrics.

    Args:
        request: Prediction request with student data

    Returns:
        Prediction with explainability
    """
    # Fallback to rule-based prediction if ML not available
    if not ML_AVAILABLE or enrollment_predictor is None:
        logger.warning("Using rule-based fallback for enrollment prediction")

        # Simple rule-based prediction
        risk_score = (
            (1 - request.gpa / 4.0) * 0.3 +
            (1 - request.attendance_rate / 100) * 0.25 +
            (1 - request.engagement_score) * 0.20 +
            (request.num_failed_courses / 5.0) * 0.15 +
            (request.previous_dropout_risk) * 0.10
        )
        risk_score = min(max(risk_score, 0.0), 1.0)

        if risk_score < 0.3 or risk_score < 0.6:
            pass
        else:
            pass

        from shared.resilience.ml_fallback import rule_based_enrollment_prediction

        # Use enhanced rule-based prediction
        student_data = {
            "gpa": request.gpa,
            "credits_enrolled": request.credits_enrolled,
            "attendance_rate": request.attendance_rate,
            "engagement_score": request.engagement_score,
            "previous_dropout_risk": request.previous_dropout_risk,
            "course_difficulty": request.course_difficulty,
            "study_hours": request.study_hours,
            "num_failed_courses": request.num_failed_courses,
        }

        prediction = rule_based_enrollment_prediction(student_data)

        return EnrollmentPredictionResponse(
            student_id=request.student_id,
            dropout_probability=prediction["dropout_probability"],
            retention_probability=prediction["retention_probability"],
            risk_level=prediction["risk_level"],
            confidence=prediction["confidence"],
            explanation=prediction["explanation"] if request.explain else None,
        )

    logger.info("Enrollment prediction request", student_id=request.student_id)

    # Prepare input data
    input_data = {
        'gpa': request.gpa,
        'credits_enrolled': request.credits_enrolled,
        'attendance_rate': request.attendance_rate,
        'engagement_score': request.engagement_score,
        'previous_dropout_risk': request.previous_dropout_risk,
        'course_difficulty': request.course_difficulty,
        'study_hours': request.study_hours,
        'num_failed_courses': request.num_failed_courses,
    }

    # Get prediction
    prediction = await enrollment_predictor.predict(input_data)

    # Get explanation if requested
    explanation = None
    if request.explain:
        explanation = await enrollment_predictor.explain(input_data, prediction)

    logger.info(
        "Prediction complete",
        student_id=request.student_id,
        dropout_prob=f"{prediction['dropout_probability']:.1%}",
        risk_level=prediction['risk_level'],
    )

    return EnrollmentPredictionResponse(
        student_id=request.student_id,
        dropout_probability=prediction['dropout_probability'],
        retention_probability=prediction['retention_probability'],
        risk_level=prediction['risk_level'],
        confidence=prediction['confidence'],
        explanation=explanation,
    )


@app.post("/api/v1/optimize/rooms", response_model=RoomOptimizationResponse)
async def optimize_room_allocation(request: RoomOptimizationRequest) -> RoomOptimizationResponse:
    """
    Optimize room allocation - REAL RL MODEL!

    Uses PPO reinforcement learning to find optimal room assignments
    that minimize energy cost and travel time while satisfying constraints.

    Args:
        request: Optimization request with sections and rooms

    Returns:
        Optimal allocation with metrics
    """
    # Fallback to rule-based allocation if ML not available
    if not ML_AVAILABLE or room_optimizer is None:
        logger.warning("Using rule-based fallback for room optimization")

        from shared.resilience.ml_fallback import rule_based_room_optimization

        # Use enhanced rule-based optimization
        request_data = {
            "sections": request.sections,
            "rooms": request.rooms,
        }

        result = rule_based_room_optimization(request_data)

        return RoomOptimizationResponse(
            allocation=result["allocation"],
            metrics=result["metrics"],
            num_sections_allocated=result["num_sections_allocated"],
            num_violations=result["num_violations"],
            explanation=result["explanation"] if request.explain else None,
        )

    logger.info(
        "Room optimization request",
        num_sections=len(request.sections),
        num_rooms=len(request.rooms),
    )

    # Prepare input data
    input_data = {
        'sections': request.sections,
        'rooms': request.rooms,
    }

    # Get optimization
    result = await room_optimizer.predict(input_data)

    # Get explanation if requested
    explanation = None
    if request.explain:
        explanation = await room_optimizer.explain(input_data, result)

    logger.info(
        "Optimization complete",
        sections_allocated=result['num_sections_allocated'],
        violations=result['num_violations'],
    )

    return RoomOptimizationResponse(
        allocation=result['allocation'],
        metrics=result['metrics'],
        num_sections_allocated=result['num_sections_allocated'],
        num_violations=result['num_violations'],
        explanation=explanation,
    )


@app.get("/api/v1/models/status")
async def get_model_status() -> dict[str, Any]:
    """Get status of loaded ML models."""
    return {
        'enrollment_predictor': {
            'loaded': enrollment_predictor is not None,
            'trained': enrollment_predictor.is_trained if enrollment_predictor else False,
            'model_name': enrollment_predictor.model_name if enrollment_predictor else None,
            'version': enrollment_predictor.version if enrollment_predictor else None,
        },
        'room_optimizer': {
            'loaded': room_optimizer is not None,
            'trained': room_optimizer.is_trained if room_optimizer else False,
            'model_name': room_optimizer.model_name if room_optimizer else None,
            'version': room_optimizer.version if room_optimizer else None,
        },
    }


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "service": "Analytics Service",
        "version": "0.1.0",
        "status": "operational",
        "features": [
            "enrollment_dropout_prediction",
            "room_usage_optimization",
            "explainable_ai",
        ],
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.analytics_service.main:app",
        host="0.0.0.0",
        port=settings.analytics_service_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )

