"""
SentinelForge — tests/integration/test_redis.py

PURPOSE:
    Integration tests for RedisClient operations, string caching, hash maps,
    and sliding window sorted sets. Supports both live Redis instances and
    mocked aioredis connections.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.db.redis_client import RedisClient


@pytest.mark.asyncio
async def test_redis_client_operations_mocked() -> None:
    """Verify RedisClient string, hash, and sorted set operations with mock."""
    client = RedisClient()

    mock_aioredis = MagicMock()
    mock_aioredis.ping = AsyncMock(return_value=True)
    mock_aioredis.get = AsyncMock(return_value="cached_val")
    mock_aioredis.set = AsyncMock(return_value=True)
    mock_aioredis.hset = AsyncMock(return_value=5)
    mock_aioredis.hgetall = AsyncMock(return_value={"txn_count_1min": "3"})
    mock_aioredis.zadd = AsyncMock(return_value=1)
    mock_aioredis.zcount = AsyncMock(return_value=4)
    mock_aioredis.zremrangebyscore = AsyncMock(return_value=2)
    mock_aioredis.aclose = AsyncMock()

    with patch("redis.asyncio.from_url", return_value=mock_aioredis):
        await client.connect()
        assert client.client is not None

        # Health
        assert await client.is_healthy() is True

        # String get/set
        await client.set("key", "val", expire_seconds=60)
        mock_aioredis.set.assert_called_once_with("key", "val", ex=60)
        assert await client.get("key") == "cached_val"

        # Hash map (feature vector storage)
        fields = {"txn_count_1min": 3, "avg_amount": 120.5}
        await client.hset("features:txn_1", fields)
        features = await client.hgetall("features:txn_1")
        assert features == {"txn_count_1min": "3"}

        # Sorted Set (velocity sliding window)
        await client.zadd("velocity:c1", {"txn_1": 1700000000.0})
        count = await client.zcount("velocity:c1", 1699999000.0, 1700000000.0)
        assert count == 4
        pruned = await client.zremrangebyscore("velocity:c1", 0.0, 1699999000.0)
        assert pruned == 2

        # Disconnect
        await client.disconnect()
        assert client.client is None
