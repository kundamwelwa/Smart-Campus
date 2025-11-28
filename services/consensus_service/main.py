"""
Consensus Service - Raft-based Distributed State Management

Provides distributed consensus for critical enrollment state using Raft algorithm.
Ensures consistency across multiple Academic Service replicas.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from shared.config import settings
from shared.consensus.raft import RaftNode

logger = structlog.get_logger(__name__)

# Global Raft node
raft_node: RaftNode | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan - initialize Raft node."""
    global raft_node

    logger.info("Starting Consensus Service")

    # Initialize Raft cluster
    # In production, cluster_nodes would come from service discovery
    cluster_nodes = [1, 2, 3]  # 3-node cluster

    raft_node = RaftNode(
        node_id=settings.raft_node_id,
        cluster_nodes=cluster_nodes,
    )

    # Start Raft protocol
    await raft_node.start()

    logger.info(
        "Raft node started",
        node_id=settings.raft_node_id,
        cluster_size=len(cluster_nodes),
    )

    yield

    # Shutdown
    if raft_node:
        await raft_node.stop()

    logger.info("Consensus Service shutdown complete")


app = FastAPI(
    title="Argos Consensus Service",
    description="Distributed consensus using Raft algorithm",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
)


class WriteRequest(BaseModel):
    """Request to write to replicated state."""

    key: str
    value: Any


class WriteResponse(BaseModel):
    """Response for write operation."""

    success: bool
    committed: bool
    leader_id: int | None


class ReadRequest(BaseModel):
    """Request to read from replicated state."""

    key: str


class ReadResponse(BaseModel):
    """Response for read operation."""

    key: str
    value: Any | None
    found: bool


@app.post("/api/v1/write", response_model=WriteResponse)
async def write_value(request: WriteRequest) -> WriteResponse:
    """
    Write value to replicated state using Raft consensus.

    This ensures the write is replicated to a majority of nodes
    before being committed.

    Args:
        request: Write request

    Returns:
        WriteResponse: Write result
    """
    if raft_node is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Raft node not initialized",
        )

    # Prepare command
    command = {
        'type': 'write',
        'key': request.key,
        'value': request.value,
    }

    # Append to Raft log (will replicate if leader)
    success = await raft_node.append_entry(command)

    if not success:
        # Not leader - return leader info
        return WriteResponse(
            success=False,
            committed=False,
            leader_id=raft_node.current_leader,
        )

    logger.info(
        "Value written via Raft",
        key=request.key,
        node_id=raft_node.node_id,
    )

    return WriteResponse(
        success=True,
        committed=True,
        leader_id=raft_node.node_id,
    )


@app.get("/api/v1/read", response_model=ReadResponse)
async def read_value(key: str) -> ReadResponse:
    """
    Read value from replicated state.

    Reads from local state machine (eventual consistency).
    For strong consistency, leader read would be required.

    Args:
        key: Key to read

    Returns:
        ReadResponse: Read result
    """
    if raft_node is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Raft node not initialized",
        )

    # Read from local state
    value = raft_node.read_state(key)

    return ReadResponse(
        key=key,
        value=value,
        found=value is not None,
    )


@app.get("/api/v1/status")
async def get_consensus_status() -> dict[str, Any]:
    """
    Get Raft cluster status.

    Returns:
        dict: Cluster state information
    """
    if raft_node is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Raft node not initialized",
        )

    return raft_node.get_state()


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "consensus_service",
        "node_id": str(settings.raft_node_id) if raft_node else "not_initialized",
        "state": raft_node.state.value if raft_node else "unknown",
    }


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "service": "Consensus Service",
        "version": "0.1.0",
        "algorithm": "Raft",
        "status": "operational",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.consensus_service.main:app",
        host="0.0.0.0",
        port=settings.consensus_service_port,
        reload=False,  # Don't reload - Raft state would be lost
        log_level=settings.log_level.lower(),
    )

