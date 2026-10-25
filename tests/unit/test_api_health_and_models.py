"""
SentinelForge — tests/unit/test_api_health_and_models.py

PURPOSE:
    Unit tests verifying health probes, Prometheus metrics scrape, and model lifecycle endpoints.
"""

import httpx
import pytest
from httpx import ASGITransport

from services.api.dependencies import create_access_token
from services.api.main import app


@pytest.mark.asyncio
async def test_health_check_endpoint() -> None:
    """Verify GET /api/v1/health returns system readiness diagnostic."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "active_model_version" in data
        assert "postgres_connected" in data
        assert "redis_connected" in data


@pytest.mark.asyncio
async def test_prometheus_metrics_scrape() -> None:
    """Verify GET /api/v1/health/metrics returns plaintext metrics exposition."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/health/metrics")
        assert resp.status_code == 200
        assert "sentinelforge" in resp.text


@pytest.mark.asyncio
async def test_model_lifecycle_endpoints() -> None:
    """Verify model listing, inspection, rollback, and drift status endpoints."""
    token = create_access_token({"sub": "admin", "role": "admin"})
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. List models
        resp = await client.get("/api/v1/models", headers=headers)
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

        # 2. Get current production champion
        resp_curr = await client.get("/api/v1/models/lightgbm_fraud/current", headers=headers)
        assert resp_curr.status_code == 200
        assert resp_curr.json()["model_name"] == "lightgbm_fraud"

        # 3. Rollback
        rollback_payload = {"target_version": "baseline_v1", "reason": "Operational safety check"}
        resp_rb = await client.post(
            "/api/v1/models/lightgbm_fraud/rollback",
            json=rollback_payload,
            headers=headers,
        )
        assert resp_rb.status_code == 200
        assert resp_rb.json()["status"] == "rolled_back"

        # 4. Drift status
        resp_drift = await client.get("/api/v1/models/drift/status", headers=headers)
        assert resp_drift.status_code == 200
        assert resp_drift.json()["model_name"] == "lightgbm_fraud"
