"""
SentinelForge — services/api/routers/health_router.py

PURPOSE:
    Provides system health checks, dependency readiness probes, and Prometheus
    metric scraping endpoints for automated container orchestration.

ARCHITECTURE POSITION:
    Layer 5 (FastAPI Gateway) → Unauthenticated monitoring endpoints.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from services.scoring.model_loader import ModelLoader, get_model_loader
from shared.config.settings import Settings, get_settings
from shared.db.postgres import PostgresClient, get_postgres_client
from shared.db.redis_client import RedisClient, get_redis_client

health_router = APIRouter(prefix="/api/v1/health", tags=["Observability & Health"])


class HealthResponse(BaseModel):
    """Platform health check diagnostic summary."""

    status: str = Field(..., description="Overall health state ('healthy' or 'degraded')")
    environment: str = Field(..., description="Runtime environment")
    active_model_version: str = Field(..., description="Current loaded scoring model version")
    postgres_connected: bool = Field(..., description="PostgreSQL database connectivity")
    redis_connected: bool = Field(..., description="Redis online feature store connectivity")


@health_router.get(
    "",
    response_model=HealthResponse,
    summary="System readiness and dependency probe",
)
async def check_health(
    settings: Annotated[Settings, Depends(get_settings)],
    postgres: Annotated[PostgresClient, Depends(get_postgres_client)],
    redis: Annotated[RedisClient, Depends(get_redis_client)],
    model_loader: Annotated[ModelLoader, Depends(get_model_loader)],
) -> HealthResponse:
    """
    Probe system readiness and active database connections.

    Returns:
        HealthResponse: Status of database, cache, and active model.
    """
    pg_ok = await postgres.is_healthy()
    redis_ok = await redis.is_healthy()

    overall_status = "healthy" if (pg_ok and redis_ok) else "degraded"

    return HealthResponse(
        status=overall_status,
        environment=settings.ENVIRONMENT,
        active_model_version=model_loader.current_version,
        postgres_connected=pg_ok,
        redis_connected=redis_ok,
    )


@health_router.get(
    "/metrics",
    summary="Scrape Prometheus observability metrics",
)
async def get_prometheus_metrics() -> Response:
    """
    Export collected Prometheus metrics in standard text format.

    Returns:
        Response: Plaintext metric exposition scraped by Prometheus container.
    """
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
