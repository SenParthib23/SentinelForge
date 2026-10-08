"""
SentinelForge — services/feature_engine/online_store.py

PURPOSE:
    High-performance online feature store client using Redis Hashes.
    Provides sub-millisecond retrieval of pre-computed feature vectors
    for synchronous real-time LightGBM scoring.

ARCHITECTURE POSITION:
    Layer 2 (Feature Engine) → Layer 3 (Synchronous Scoring Path).

KEY PATTERN:
    Key: `features:{transaction_id}`
    Data Structure: Redis Hash
    TTL: Configurable (default 86,400 seconds / 24 hours)

INTERVIEW TALKING POINT:
    "We use Redis Hashes for online feature storage because HGETALL executes
    in under 0.5ms over local loopback / VPC. Storing individual fields in a Hash
    rather than a single JSON string allows us to do zero-copy partial field updates
    if a late-arriving signal arrives before authorization finishes."
"""

from typing import Optional

from shared.config.settings import Settings, get_settings
from shared.db.redis_client import RedisClient, get_redis_client
from shared.logging.logger import get_logger
from shared.models.feature_vector import FeatureVector

logger = get_logger("feature_engine.online_store")


class OnlineFeatureStore:
    """
    Manages saving and fetching feature vectors to and from the Redis online store.
    """

    def __init__(
        self,
        redis_client: Optional[RedisClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize online feature store.

        Args:
            redis_client: RedisClient instance.
            settings: Settings instance.
        """
        self.redis = redis_client or get_redis_client()
        self.settings = settings or get_settings()
        self._local_cache: dict[str, FeatureVector] = {}

    def _format_key(self, transaction_id: str) -> str:
        """Generate standardized Redis key name."""
        return f"features:{transaction_id}"

    async def save_feature_vector(
        self,
        feature_vector: FeatureVector,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """
        Persist a computed FeatureVector into the Redis online store with TTL.

        Args:
            feature_vector: Validated FeatureVector.
            ttl_seconds: Optional TTL in seconds (defaults to settings.FEATURE_TTL_SECONDS).
        """
        key = self._format_key(feature_vector.transaction_id)
        ttl = ttl_seconds or self.settings.FEATURE_TTL_SECONDS

        # Convert vector to dictionary for Redis hash storage
        payload = feature_vector.model_dump()
        payload["computed_at"] = feature_vector.computed_at.isoformat()

        try:
            if self.redis.client is not None or await self.redis.ping():
                await self.redis.hset(key, payload)
                await self.redis.expire(key, ttl)
                return
        except Exception as exc:
            logger.debug(
                "Redis online store save using local memory fallback",
                error=str(exc),
            )

        # In-memory backup
        self._local_cache[feature_vector.transaction_id] = feature_vector

    async def get_feature_vector(
        self, transaction_id: str
    ) -> Optional[FeatureVector]:
        """
        Retrieve FeatureVector by transaction ID in sub-millisecond latency.

        Args:
            transaction_id: Unique transaction ID.

        Returns:
            Optional[FeatureVector]: FeatureVector if found, None otherwise.
        """
        key = self._format_key(transaction_id)

        try:
            if self.redis.client is not None or await self.redis.ping():
                data = await self.redis.hgetall(key)
                if data:
                    # Coerce string values back to correct types
                    return FeatureVector(
                        transaction_id=data["transaction_id"],
                        customer_id=data["customer_id"],
                        txn_count_1min=int(data.get("txn_count_1min", 1)),
                        txn_count_5min=int(data.get("txn_count_5min", 1)),
                        txn_count_1hr=int(data.get("txn_count_1hr", 1)),
                        txn_count_24hr=int(data.get("txn_count_24hr", 1)),
                        avg_amount_1hr=float(data.get("avg_amount_1hr", 0.0)),
                        max_amount_24hr=float(data.get("max_amount_24hr", 0.0)),
                        amount_zscore=float(data.get("amount_zscore", 0.0)),
                        unique_devices_1hr=int(data.get("unique_devices_1hr", 1)),
                        device_txn_count=int(data.get("device_txn_count", 1)),
                        is_new_device=data.get("is_new_device", "False").lower() == "true",
                        merchant_avg_amount=float(data.get("merchant_avg_amount", 0.0)),
                        merchant_fraud_rate=float(data.get("merchant_fraud_rate", 0.0)),
                        is_new_merchant=data.get("is_new_merchant", "False").lower() == "true",
                        distance_from_last_txn=float(data.get("distance_from_last_txn", 0.0)),
                        is_international=data.get("is_international", "False").lower() == "true",
                        hour_of_day=int(data.get("hour_of_day", 12)),
                        is_weekend=data.get("is_weekend", "False").lower() == "true",
                        feature_version=data.get("feature_version", "1.0.0"),
                    )
        except Exception as exc:
            logger.debug(
                "Redis online store get using local memory fallback",
                error=str(exc),
            )

        return self._local_cache.get(transaction_id)
