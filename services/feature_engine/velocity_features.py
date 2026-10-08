"""
SentinelForge — services/feature_engine/velocity_features.py

PURPOSE:
    Computes real-time transaction velocity features (transaction burst counters
    within sliding time windows) for fraud detection scoring.

ARCHITECTURE POSITION:
    Layer 2 (Feature Engine) → consumed by Layer 3 (Scoring).

HOW IT WORKS:
    Uses Redis Sorted Sets with transaction timestamps as scores.
    ZADD records each event with timestamp, ZCOUNT calculates events within
    `[current_time - window_size, current_time]`, and ZREMRANGEBYSCORE prunes
    stale entries older than the longest active window (24 hours).

WINDOW SIZES:
    - 1 minute (60s)
    - 5 minutes (300s)
    - 1 hour (3600s)
    - 24 hours (86400s)

WHY REDIS SORTED SETS:
    - O(log N) insertion and range queries.
    - Native sliding window semantics without maintaining in-memory worker state.
    - Shared across horizontally scaled Faust/Kafka streaming workers.

INTERVIEW TALKING POINT:
    "For velocity features, I used Redis Sorted Sets because they give O(log N)
    sliding window queries natively. The alternative was maintaining in-memory
    counters in the Faust worker, but those don't survive restarts and can't be
    shared across worker replicas without distributed state locks."
"""

from collections import defaultdict
from typing import Optional

from shared.db.redis_client import RedisClient, get_redis_client
from shared.logging.logger import get_logger

logger = get_logger("features.velocity")


class VelocityFeatureComputer:
    """
    Computes transaction velocity features using Redis Sorted Sets or in-memory fallback.
    """

    def __init__(self, redis_client: Optional[RedisClient] = None) -> None:
        """
        Initialize VelocityFeatureComputer.

        Args:
            redis_client: Optional Redis client; defaults to global singleton.
        """
        self.redis_client = redis_client or get_redis_client()
        # In-memory backup structures for local tests/isolated execution
        self._local_customer_history: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self._local_device_history: dict[str, list[tuple[str, float]]] = defaultdict(list)

    async def record_and_compute(
        self,
        customer_id: str,
        device_fingerprint: str,
        transaction_id: str,
        timestamp_epoch: float,
    ) -> dict[str, int]:
        """
        Record current transaction and return velocity counters across all windows.

        Args:
            customer_id: Unique customer identifier.
            device_fingerprint: Device hardware/browser fingerprint.
            transaction_id: Unique transaction reference.
            timestamp_epoch: Unix epoch timestamp in seconds.

        Returns:
            dict[str, int]: Dictionary containing:
                - txn_count_1min: int
                - txn_count_5min: int
                - txn_count_1hr: int
                - txn_count_24hr: int
                - device_txn_count: int
        """
        cust_key = f"velocity:cust:{customer_id}"
        dev_key = f"velocity:dev:{device_fingerprint}"

        try:
            # Check if Redis is connected
            if self.redis_client.client is not None or await self.redis_client.ping():
                # Add current transaction to customer and device sorted sets
                await self.redis_client.zadd(cust_key, {transaction_id: timestamp_epoch})
                await self.redis_client.zadd(dev_key, {transaction_id: timestamp_epoch})

                # Set 24h + 1hr TTL on keys so Redis frees memory automatically
                await self.redis_client.expire(cust_key, 90000)
                await self.redis_client.expire(dev_key, 7200)

                # Query sliding windows for customer
                count_1min = await self.redis_client.zcount(
                    cust_key, timestamp_epoch - 60, timestamp_epoch
                )
                count_5min = await self.redis_client.zcount(
                    cust_key, timestamp_epoch - 300, timestamp_epoch
                )
                count_1hr = await self.redis_client.zcount(
                    cust_key, timestamp_epoch - 3600, timestamp_epoch
                )
                count_24hr = await self.redis_client.zcount(
                    cust_key, timestamp_epoch - 86400, timestamp_epoch
                )

                # Query sliding window for device (1hr)
                dev_count_1hr = await self.redis_client.zcount(
                    dev_key, timestamp_epoch - 3600, timestamp_epoch
                )

                # Asynchronously prune stale entries older than 24h
                await self.redis_client.zremrangebyscore(
                    cust_key, 0.0, timestamp_epoch - 86400
                )
                await self.redis_client.zremrangebyscore(
                    dev_key, 0.0, timestamp_epoch - 3600
                )

                return {
                    "txn_count_1min": max(1, count_1min),
                    "txn_count_5min": max(1, count_5min),
                    "txn_count_1hr": max(1, count_1hr),
                    "txn_count_24hr": max(1, count_24hr),
                    "device_txn_count": max(1, dev_count_1hr),
                }

        except Exception as exc:
            logger.debug(
                "Redis velocity calculation using fallback in-memory store",
                error=str(exc),
            )

        # In-memory fallback computation
        self._local_customer_history[customer_id].append((transaction_id, timestamp_epoch))
        self._local_device_history[device_fingerprint].append((transaction_id, timestamp_epoch))

        # Prune in-memory older than 24h
        cutoff_24hr = timestamp_epoch - 86400
        self._local_customer_history[customer_id] = [
            (t, ts) for t, ts in self._local_customer_history[customer_id] if ts >= cutoff_24hr
        ]
        self._local_device_history[device_fingerprint] = [
            (t, ts) for t, ts in self._local_device_history[device_fingerprint] if ts >= (timestamp_epoch - 3600)
        ]

        cust_ts = [ts for _, ts in self._local_customer_history[customer_id]]
        dev_ts = [ts for _, ts in self._local_device_history[device_fingerprint]]

        return {
            "txn_count_1min": max(1, sum(1 for ts in cust_ts if ts >= timestamp_epoch - 60)),
            "txn_count_5min": max(1, sum(1 for ts in cust_ts if ts >= timestamp_epoch - 300)),
            "txn_count_1hr": max(1, sum(1 for ts in cust_ts if ts >= timestamp_epoch - 3600)),
            "txn_count_24hr": max(1, len(cust_ts)),
            "device_txn_count": max(1, len(dev_ts)),
        }
