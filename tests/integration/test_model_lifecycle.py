"""
SentinelForge — tests/integration/test_model_lifecycle.py

PURPOSE:
    Integration tests for Layer 10 & 11 API endpoints:
    Canary traffic splitting, automated status checks, emergency rollback, and drift monitoring.
"""

import httpx
import pytest
from httpx import ASGITransport

from services.api.dependencies import create_access_token
from services.api.main import app


@pytest.mark.asyncio
async def test_canary_deployment_and_rollback_api() -> None:
    """Verify canary split adjustment, health monitoring, and rollback endpoints."""
    token = create_access_token({"sub": "admin_user", "role": "admin"})
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Configure Canary Split
        payload = {
            "canary_percentage": 20.0,
            "canary_version": "candidate_v2",
            "shadow_mode": False,
        }
        resp_split = await client.post("/api/v1/models/canary/split", json=payload, headers=headers)
        assert resp_split.status_code == 200
        split_data = resp_split.json()
        assert split_data["status"] == "updated"
        assert split_data["configuration"]["canary_percentage"] == 20.0

        # 2. Get Canary Status
        resp_status = await client.get("/api/v1/models/canary/status", headers=headers)
        assert resp_status.status_code == 200
        status_data = resp_status.json()
        assert status_data["canary_active"] is True
        assert status_data["canary_percentage"] == 20.0

        # 3. Emergency Rollback
        rollback_payload = {
            "target_version": "baseline_v1",
            "reason": "Integration test emergency rollback trigger",
        }
        resp_rb = await client.post(
            "/api/v1/models/lightgbm_fraud/rollback", json=rollback_payload, headers=headers
        )
        assert resp_rb.status_code == 200
        assert resp_rb.json()["status"] == "rolled_back"

        # 4. Verify Canary was reset to 0%
        resp_after = await client.get("/api/v1/models/canary/status", headers=headers)
        assert resp_after.status_code == 200
        assert resp_after.json()["canary_percentage"] == 0.0

        # 5. Check Drift Status
        resp_drift = await client.get("/api/v1/models/drift/status", headers=headers)
        assert resp_drift.status_code == 200
        assert resp_drift.json()["model_name"] == "lightgbm_fraud"
