"""
Health check endpoint with circuit breaker status.
"""

from fastapi import APIRouter

from shared.resilience.circuit_breaker import circuit_breaker_manager

router = APIRouter()


@router.get("/circuit-breakers")
async def get_circuit_breaker_status():
    """
    Get status of all circuit breakers.

    Returns:
        Dictionary of circuit breaker statistics
    """
    return circuit_breaker_manager.get_all_stats()
