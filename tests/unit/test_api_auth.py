"""
SentinelForge — tests/unit/test_api_auth.py

PURPOSE:
    Unit tests verifying JWT authentication, login token issuance,
    and protected endpoint authorization gates.
"""

import httpx
import pytest
from httpx import ASGITransport

from services.api.main import app


@pytest.mark.asyncio
async def test_login_valid_credentials() -> None:
    """Verify login with valid credentials returns signed JWT access token."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "sentinelforge2026"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["username"] == "admin"
        assert data["role"] == "admin"


@pytest.mark.asyncio
async def test_login_invalid_credentials() -> None:
    """Verify login with invalid credentials returns HTTP 401."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "wrong_password_123"},
        )

        assert resp.status_code == 401
        assert "Invalid username or password" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_protected_endpoint_without_token() -> None:
    """Verify protected endpoints reject requests lacking Authorization header."""
    async with httpx.AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/cases")
        assert resp.status_code == 401
