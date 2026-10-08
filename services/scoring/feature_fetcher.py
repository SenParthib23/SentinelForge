"""
SentinelForge — services/scoring/feature_fetcher.py

PURPOSE:
    Low-latency (<1ms SLA) online feature vector retrieval from Redis.
    Bridges the real-time scoring engine with fresh behavioral features
    computed by the streaming engine.

ARCHITECTURE POSITION:
    Layer 3 (Real-Time Scoring) → Subsystem for online feature lookups.

OBSERVABILITY:
    Instruments retrieval duration using `track_feature_fetch_latency`
    to monitor p99 Redis fetch times against the auth path latency budget.
"""

import time
from typing import Optional

from services.feature_engine.online_store import OnlineFeatureStore
from shared.logging.logger import get_logger
from shared.metrics.prometheus import track_feature_fetch_latency
from shared.models.feature_vector import FeatureVector
from shared.models.transaction import TransactionEvent

logger = get_logger("scoring.feature_fetcher")


class FeatureFetcher:
    """
    Retrieves fresh online feature vectors with sub-millisecond latency.
    """

    def __init__(self, online_store: Optional[OnlineFeatureStore] = None) -> None:
        """
        Initialize FeatureFetcher.

        Args:
            online_store: OnlineFeatureStore instance.
        """
        self.online_store = online_store or OnlineFeatureStore()

    async def fetch_or_fallback(
        self,
        transaction_id: str,
        event: Optional[TransactionEvent] = None,
    ) -> tuple[FeatureVector, float]:
        """
        Fetch FeatureVector from Redis, measuring latency and providing safe defaults.

        Args:
            transaction_id: Unique transaction reference.
            event: Optional raw event to synthesize default features if Redis misses.

        Returns:
            tuple[FeatureVector, float]: (FeatureVector, fetch_latency_ms)
        """
        start = time.perf_counter()

        with track_feature_fetch_latency():
            vector = await self.online_store.get_feature_vector(transaction_id)

        fetch_ms = (time.perf_counter() - start) * 1000.0

        if vector is not None:
            return vector, fetch_ms

        # Fallback default feature vector
        logger.warning(
            "Feature vector cache miss in online store; using fallback vector",
            transaction_id=transaction_id,
        )
        fallback_vector = FeatureVector(
            transaction_id=transaction_id,
            customer_id=event.customer_id if event else "unknown_customer",
            txn_count_1min=1,
            txn_count_5min=1,
            txn_count_1hr=1,
            txn_count_24hr=1,
            avg_amount_1hr=float(event.amount) if event else 500.0,
            max_amount_24hr=float(event.amount) if event else 500.0,
            amount_zscore=0.0,
            unique_devices_1hr=1,
            device_txn_count=1,
            is_new_device=False,
            merchant_avg_amount=500.0,
            merchant_fraud_rate=0.015,
            is_new_merchant=False,
            distance_from_last_txn=0.0,
            is_international=event.is_international if event else False,
            hour_of_day=event.timestamp.hour if event else 12,
            is_weekend=bool(event.timestamp.weekday() >= 5) if event else False,
        )

        return fallback_vector, fetch_ms
