"""
SentinelForge — tests/integration/test_postgres.py

PURPOSE:
    Integration tests for PostgresClient connection management, query execution,
    and health probing. Supports both live PostgreSQL instances (when available)
    and mocked asyncpg connection pools.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.config.settings import Settings
from shared.db.postgres import PostgresClient


@pytest.mark.asyncio
async def test_postgres_client_connect_and_health_mocked() -> None:
    """Verify PostgresClient initialization, health probe, and disconnect under mock."""
    client = PostgresClient()

    # Mock asyncpg pool
    mock_pool = MagicMock()
    mock_pool.close = AsyncMock()

    mock_conn = MagicMock()
    mock_conn.fetchval = AsyncMock(return_value=1)
    mock_conn.execute = AsyncMock(return_value="CREATE TABLE")

    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_acquire.__aexit__ = AsyncMock(return_value=None)
    mock_pool.acquire.return_value = mock_acquire

    with patch("asyncpg.create_pool", AsyncMock(return_value=mock_pool)):
        await client.connect()
        assert client.pool is not None

        # Health probe
        healthy = await client.is_healthy()
        assert healthy is True

        # Execute query
        result = await client.execute("CREATE TABLE test (id INT);")
        assert result == "CREATE TABLE"

        # Disconnect
        await client.disconnect()
        assert client.pool is None


@pytest.mark.asyncio
async def test_postgres_client_handles_connection_failure() -> None:
    """Verify PostgresClient raises and logs on connection failure."""
    client = PostgresClient()

    with patch("asyncpg.create_pool", AsyncMock(side_effect=ConnectionRefusedError("Refused"))):
        with pytest.raises(ConnectionRefusedError):
            await client.connect()
