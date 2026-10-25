"""
SentinelForge — tests/unit/test_api_scoring.py

PURPOSE:
    Unit tests verifying real-time scoring endpoint `POST /api/v1/score`.
"""

import httpx
import pytest
from httpx import ASGITransport

from services.api.dependencies import create_access_token
from services.api.main import app
from shared.models.transaction import TransactionEvent


@pytest.mark.asyncio
async def test_score_endpoint_success(sample_transaction_event: TransactionEvent) -> None:
    """Verify POST /api/v1/score returns valid prediction and latencies."""
    token = create_access_token({"sub": "analyst_jane", "role": "analyst"})
    headers = {"Authorization": f"Bearer {token}"}

    payload = sample_transaction_event.model_dump()
    payload["timestamp"] = sample_transaction_event.timestamp.isoformat()
    payload["amount"] = float(sample_transaction_event.amount)

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/score", json=payload, headers=headers)

        assert resp.status_code == 200
        data = resp.json()
        assert data["transaction_id"] == sample_transaction_event.transaction_id
        assert 0.0 <= data["risk_score"] <= 1.0
        assert data["risk_label"] in ["low", "medium", "high", "critical"]
        assert "scoring_latency_ms" in data
        assert "feature_fetch_latency_ms" in data
