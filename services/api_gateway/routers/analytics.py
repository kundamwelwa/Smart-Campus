"""Analytics Router - ML predictions and recommendations"""
import httpx
import structlog
from fastapi import APIRouter, HTTPException, Request

from shared.config import settings

router = APIRouter()
logger = structlog.get_logger(__name__)

ANALYTICS_SERVICE_URL = f"http://localhost:{settings.analytics_service_port}/api/v1"


@router.get("/recommendations")
async def get_recommendations() -> dict[str, str]:
    """Get personalized recommendations."""
    return {"message": "Recommendations will be implemented"}


@router.post("/predict/enrollment")
async def predict_enrollment(request: Request):
    """
    Predict student enrollment/dropout probability.

    Proxies to Analytics Service - REAL ML MODEL!
    """
    try:
        headers = {"Content-Type": "application/json"}
        auth_header = request.headers.get("Authorization")
        if auth_header:
            headers["Authorization"] = auth_header

        json_data = await request.json()

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ANALYTICS_SERVICE_URL}/predict/enrollment",
                headers=headers,
                json=json_data,
            )
            response.raise_for_status()
            return response.json()

    except httpx.RequestError as e:
        logger.error("Failed to proxy enrollment prediction", error=str(e))
        raise HTTPException(status_code=503, detail="Analytics Service unavailable")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except Exception as e:
        logger.error("Unexpected error proxying enrollment prediction", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

