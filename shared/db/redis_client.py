"""
SentinelForge — shared/db/redis_client.py

PURPOSE:
    Asynchronous Redis client wrapper built on `redis.asyncio` powering the online
    feature store and real-time sliding-window velocity aggregations.

ARCHITECTURE POSITION:
    Layer 0 (Foundation) → Online feature lookup (<1ms read SLA) for Real-Time Scorer
    (Layer 3) and sliding window state for Feature Engine (Layer 2).

HOW IT WORKS:
    Maintains a managed async connection pool to Redis. Provides typed helper methods
    for key-value storage, hash mappings (for complete feature vectors), sorted sets
    (for timestamp-scored velocity sliding windows), and batched pipeline execution.

WHY REDIS FOR ONLINE FEATURE STORE:
    1. Sub-millisecond latency: Memory-resident key-value access fits inside the 150ms auth SLA.
    2. Native Data Structures: Sorted Sets (ZADD, ZCOUNT, ZREMRANGEBYSCORE) provide O(log N)
       sliding time windows without custom database query overhead.
    3. Automatic TTL: Features expire naturally after 24 hours, preventing uncontrolled RAM growth.

INTERVIEW TALKING POINT:
    "We use Redis Sorted Sets for real-time velocity counters. Each incoming transaction
    adds an entry with its epoch timestamp as the score. Counting transactions within
    a 5-minute sliding window is a single ZCOUNT query with O(log N) complexity,
    executing in under 0.5ms. Expired entries are pruned atomically using ZREMRANGEBYSCORE."
"""

from typing import Any, Optional

import redis.asyncio as aioredis
from redis.asyncio.client import Pipeline

from shared.config.settings import Settings, get_settings
from shared.logging.logger import get_logger

logger = get_logger("db.redis")


class RedisClient:
    """
    Asynchronous Redis client wrapper for online feature store and rate limiting.

    Attributes:
        settings: Application configuration containing Redis host/port/auth.
        client: The underlying aioredis.Redis connection instance.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize RedisClient with application settings.

        Args:
            settings: Optional Settings instance; uses get_settings() if omitted.
        """
        self.settings = settings or get_settings()
        self.client: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        """
        Establish connection pool to Redis.
        """
        if self.client is not None:
            return

        try:
            logger.info(
                "Connecting to Redis online feature store",
                host=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                db=self.settings.REDIS_DB,
            )
            self.client = aioredis.from_url(
                self.settings.REDIS_URL,
                decode_responses=True,
                max_connections=50,
            )
            logger.info("Redis client connected successfully")
        except Exception as exc:
            logger.error("Failed to connect to Redis", error=str(exc))
            raise

    async def disconnect(self) -> None:
        """
        Gracefully close Redis connection pool.
        """
        if self.client is not None:
            logger.info("Closing Redis connection")
            await self.client.aclose()
            self.client = None

    async def ping(self) -> bool:
        """
        Send PING command to verify Redis connectivity.

        Returns:
            bool: True if PONG response received, False otherwise.
        """
        if self.client is None:
            return False
        try:
            return bool(await self.client.ping())
        except Exception as exc:
            logger.warning("Redis ping probe failed", error=str(exc))
            return False

    async def is_healthy(self) -> bool:
        """
        Health probe verifying active Redis responsiveness.

        Returns:
            bool: True if Redis is reachable and responsive.
        """
        return await self.ping()

    async def get(self, key: str) -> Optional[str]:
        """
        Retrieve value for a string key.

        Args:
            key: Redis key name.

        Returns:
            Optional[str]: Stored string or None if missing.
        """
        if self.client is None:
            await self.connect()
        assert self.client is not None
        return await self.client.get(key)

    async def set(
        self, key: str, value: str, expire_seconds: Optional[int] = None
    ) -> bool:
        """
        Set string key with optional time-to-live.

        Args:
            key: Redis key name.
            value: String payload to store.
            expire_seconds: Optional TTL in seconds.

        Returns:
            bool: True on success.
        """
        if self.client is None:
            await self.connect()
        assert self.client is not None
        return bool(await self.client.set(key, value, ex=expire_seconds))

    async def hset(self, key: str, mapping: dict[str, Any]) -> int:
        """
        Store multiple fields in a Redis Hash.

        Args:
            key: Hash key name.
            mapping: Key-value attributes representing feature vector fields.

        Returns:
            int: Number of fields added.
        """
        if self.client is None:
            await self.connect()
        assert self.client is not None
        # Convert non-string primitives to strings
        serialized_mapping = {k: str(v) for k, v in mapping.items()}
        return await self.client.hset(key, mapping=serialized_mapping)

    async def hget(self, key: str, field: str) -> Optional[str]:
        """
        Retrieve single field from a Redis Hash.

        Args:
            key: Hash key name.
            field: Specific feature name.

        Returns:
            Optional[str]: Field value or None.
        """
        if self.client is None:
            await self.connect()
        assert self.client is not None
        return await self.client.hget(key, field)

    async def hgetall(self, key: str) -> dict[str, str]:
        """
        Retrieve all fields and values from a Redis Hash.

        Args:
            key: Hash key name.

        Returns:
            dict[str, str]: All key-value features stored at this key.
        """
        if self.client is None:
            await self.connect()
        assert self.client is not None
        return await self.client.hgetall(key)

    async def expire(self, key: str, seconds: int) -> bool:
        """
        Set time-to-live on an existing key.

        Args:
            key: Key name.
            seconds: Expiration window in seconds.

        Returns:
            bool: True if timeout was set, False if key does not exist.
        """
        if self.client is None:
            await self.connect()
        assert self.client is not None
        return bool(await self.client.expire(key, seconds))

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        """
        Add elements to a Sorted Set with timestamp scores for sliding window calculations.

        Args:
            key: Sorted set key (e.g. `velocity:customer:{id}`).
            mapping: Member to score (epoch timestamp) mapping.

        Returns:
            int: Number of elements added.
        """
        if self.client is None:
            await self.connect()
        assert self.client is not None
        return await self.client.zadd(key, mapping)

    async def zcount(self, key: str, min_score: float, max_score: float) -> int:
        """
        Count entries in a Sorted Set within a score range [min_score, max_score].

        Args:
            key: Sorted set key.
            min_score: Lower bound epoch timestamp.
            max_score: Upper bound epoch timestamp.

        Returns:
            int: Number of elements in window.
        """
        if self.client is None:
            await self.connect()
        assert self.client is not None
        return await self.client.zcount(key, min_score, max_score)

    async def zremrangebyscore(
        self, key: str, min_score: float, max_score: float
    ) -> int:
        """
        Prune expired entries from a sliding window Sorted Set.

        Args:
            key: Sorted set key.
            min_score: Start of pruning range (often -inf or 0).
            max_score: Upper cutoff threshold (current_time - window_size).

        Returns:
            int: Number of pruned members.
        """
        if self.client is None:
            await self.connect()
        assert self.client is not None
        return await self.client.zremrangebyscore(key, min_score, max_score)

    def pipeline(self) -> Pipeline:
        """
        Create a Redis transactional pipeline for atomic batched commands.

        Returns:
            Pipeline: Async Redis pipeline instance.
        """
        if self.client is None:
            raise RuntimeError("Redis client must be connected before creating a pipeline")
        return self.client.pipeline(transaction=True)


_redis_client: Optional[RedisClient] = None


def get_redis_client() -> RedisClient:
    """
    Retrieve or create global RedisClient singleton instance.

    Returns:
        RedisClient: Singleton Redis client instance.
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client
