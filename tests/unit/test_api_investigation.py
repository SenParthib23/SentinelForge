"""
SentinelForge — tests/unit/test_api_investigation.py

PURPOSE:
    Unit tests verifying investigation case listing, details, and analyst feedback endpoints.
"""

from uuid import uuid4

import httpx
import pytest
from httpx import ASGITransport

from services.api.dependencies import create_access_token, get_case_builder
from services.api.main import app
from shared.models.fraud_case import AnalystDecision, CaseStatus, FraudCase


@pytest.mark.asyncio
async def test_cases_listing_and_feedback() -> None:
    """Verify case retrieval and human adjudication submission."""
    token = create_access_token({"sub": "analyst_jane", "role": "analyst"})
    headers = {"Authorization": f"Bearer {token}"}

    case_builder = get_case_builder()
    test_txn_id = f"txn_case_{uuid4().hex[:8]}"

    # Seed a case into cache
    case = await case_builder.create_and_persist_case(
        transaction_id=test_txn_id,
        risk_score=0.88,
        context_bundle={"transaction_id": test_txn_id, "amount": 15000.0},
    )

    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. List cases
        resp = await client.get("/api/v1/cases", headers=headers)
        assert resp.status_code == 200
        cases = resp.json()
        assert len(cases) >= 1

        # 2. Get case detail
        resp_detail = await client.get(f"/api/v1/cases/{test_txn_id}", headers=headers)
        assert resp_detail.status_code == 200
        assert resp_detail.json()["transaction_id"] == test_txn_id

        # 3. Submit feedback
        feedback_payload = {
            "analyst_id": "analyst_jane",
            "decision": "rejected",
            "notes": "Confirmed credential stuffing attack from novel device",
        }
        resp_fb = await client.post(
            f"/api/v1/cases/{test_txn_id}/feedback",
            json=feedback_payload,
            headers=headers,
        )
        assert resp_fb.status_code == 200
        fb_data = resp_fb.json()
        assert fb_data["status"] == "resolved"
        assert fb_data["analyst_decision"] == "rejected"


@pytest.mark.asyncio
async def test_dashboard_serving() -> None:
    """Verify static dashboard HTML is served at root and /dashboard."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp_root = await client.get("/")
        assert resp_root.status_code == 200
        assert "SentinelForge" in resp_root.text
        assert "Analyst Cockpit" in resp_root.text

        resp_dash = await client.get("/dashboard")
        assert resp_dash.status_code == 200
        assert "SentinelForge" in resp_dash.text


@pytest.mark.asyncio
async def test_sse_streamer_event_dispatch() -> None:
    """Verify CaseStreamer pub/sub yields initial ping and broadcasted event."""
    from services.dashboard.sse_streamer import CaseStreamer

    streamer = CaseStreamer()
    gen = streamer.subscribe(case_id="test_case_123")

    # 1. First event is connection greeting
    init_event = await anext(gen)
    assert "connected" in init_event
    assert "test_case_123" in init_event

    # 2. Broadcast event
    await streamer.broadcast_event(
        event_type="test_alert",
        payload={"risk_score": 0.95},
        case_id="test_case_123",
    )

    broadcasted = await anext(gen)
    assert "test_alert" in broadcasted
    assert "0.95" in broadcasted


