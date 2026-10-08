"""
SentinelForge — services/scoring/scorer.py

PURPOSE:
    The core synchronous fraud scoring engine.
    Executes sub-10ms LightGBM inference, formats tabular feature arrays,
    evaluates 4-tier risk routing, tracks Prometheus metrics, and logs predictions.

ARCHITECTURE POSITION:
    Layer 3 (Synchronous Scoring Path) → Directly on the payment authorization SLA (<150ms).

LATENCY BUDGET:
    - Redis Feature Fetch: < 1.0 ms
    - Feature Vector Alignment: < 0.2 ms
    - LightGBM Inference: < 6.0 ms
    - Risk Evaluation & Routing: < 0.5 ms
    - Total Scoring Path: < 8.0 ms (Well within the 150ms hard gateway budget)

INTERVIEW TALKING POINT:
    "We convert the Pydantic FeatureVector directly into a contiguous C-aligned
    NumPy float array following an immutable column ordering. LightGBM's C++ inference
    engine executes on this array in under 5ms, avoiding Python dictionary lookup overhead."
"""

import asyncio
import time
from typing import Optional

import numpy as np

from services.scoring.feature_fetcher import FeatureFetcher
from services.scoring.model_loader import ModelLoader, get_model_loader
from services.scoring.threshold_router import ThresholdRouter
from shared.config.settings import Settings, get_settings
from shared.db.postgres import PostgresClient, get_postgres_client
from shared.logging.logger import get_logger
from shared.metrics.prometheus import (
    SCORING_REQUESTS_TOTAL,
    track_scoring_latency,
)
from shared.models.feature_vector import FeatureVector
from shared.models.fraud_case import FraudScore
from shared.models.transaction import TransactionEvent

logger = get_logger("scoring.scorer")

# Exact column ordering expected by LightGBM model
FEATURE_COLUMN_ORDER = [
    "txn_count_1min",
    "txn_count_5min",
    "txn_count_1hr",
    "txn_count_24hr",
    "avg_amount_1hr",
    "max_amount_24hr",
    "amount_zscore",
    "unique_devices_1hr",
    "device_txn_count",
    "is_new_device",
    "merchant_avg_amount",
    "merchant_fraud_rate",
    "is_new_merchant",
    "distance_from_last_txn",
    "is_international",
    "hour_of_day",
    "is_weekend",
]


class FraudScorer:
    """
    Synchronous LightGBM fraud prediction engine.
    """

    def __init__(
        self,
        model_loader: Optional[ModelLoader] = None,
        feature_fetcher: Optional[FeatureFetcher] = None,
        threshold_router: Optional[ThresholdRouter] = None,
        postgres_client: Optional[PostgresClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize FraudScorer with dependencies."""
        self.settings = settings or get_settings()
        self.model_loader = model_loader or get_model_loader()
        self.feature_fetcher = feature_fetcher or FeatureFetcher()
        self.threshold_router = threshold_router or ThresholdRouter(settings=self.settings)
        self.postgres = postgres_client or get_postgres_client()

    def _prepare_feature_array(self, vector: FeatureVector) -> np.ndarray:
        """
        Convert FeatureVector to ordered 2D float array matching LightGBM training schema.

        Args:
            vector: FeatureVector instance.

        Returns:
            np.ndarray: Shape (1, 17) contiguous numpy array.
        """
        d = vector.to_model_input_dict()
        row = [float(d.get(col, 0.0)) for col in FEATURE_COLUMN_ORDER]
        return np.array([row], dtype=np.float32)

    async def score_transaction(
        self,
        transaction_id: str,
        event: Optional[TransactionEvent] = None,
    ) -> FraudScore:
        """
        Execute synchronous sub-10ms fraud scoring decision.

        Args:
            transaction_id: Unique transaction reference.
            event: Optional raw event payload.

        Returns:
            FraudScore: Complete prediction, latency, and routing metadata.
        """
        total_start = time.perf_counter()

        # 1. Fetch online features from Redis (<1ms)
        vector, fetch_ms = await self.feature_fetcher.fetch_or_fallback(
            transaction_id, event=event
        )

        # 2. Prepare array & get loaded Booster
        X = self._prepare_feature_array(vector)
        model = self.model_loader.load_model()
        model_version = self.model_loader.current_version

        # 3. Execute LightGBM inference (<6ms) with Prometheus instrumentation
        score_start = time.perf_counter()
        with track_scoring_latency(model_version=model_version):
            raw_pred = model.predict(X)

        score_ms = (time.perf_counter() - score_start) * 1000.0

        # Handle prediction output format
        if isinstance(raw_pred, (list, np.ndarray)):
            pred_score = float(raw_pred[0])
        else:
            pred_score = float(raw_pred)

        # Clip probability to [0.0, 1.0]
        risk_score = round(max(0.0, min(1.0, pred_score)), 4)

        # 4. Evaluate operational risk tier & route
        risk_label, was_flagged, threshold_used = self.threshold_router.evaluate_risk(
            risk_score
        )

        # 5. Track Prometheus Request Counter
        SCORING_REQUESTS_TOTAL.labels(
            risk_label=risk_label.value, model_version=model_version
        ).inc()

        # 6. If flagged, asynchronously route to Kafka topic (off critical path)
        if was_flagged:
            asyncio.create_task(
                self.threshold_router.route_flagged_case(
                    transaction_id=transaction_id,
                    risk_score=risk_score,
                    risk_label=risk_label,
                    feature_dict=vector.to_model_input_dict(),
                )
            )

        fraud_score = FraudScore(
            transaction_id=transaction_id,
            model_version=model_version,
            risk_score=risk_score,
            risk_label=risk_label,
            threshold_used=threshold_used,
            was_flagged=was_flagged,
            scoring_latency_ms=round(score_ms, 2),
            feature_fetch_latency_ms=round(fetch_ms, 2),
        )

        # 7. Asynchronously persist score record to PostgreSQL audit log
        asyncio.create_task(self._persist_score_log(fraud_score))

        return fraud_score

    async def _persist_score_log(self, score: FraudScore) -> None:
        """Persist prediction audit record to PostgreSQL."""
        query = """
        INSERT INTO fraud_scores (
            transaction_id, model_version, risk_score, risk_label,
            threshold_used, was_flagged, scoring_latency_ms, feature_fetch_latency_ms
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8);
        """
        try:
            if self.postgres.pool is not None or await self.postgres.is_healthy():
                await self.postgres.execute(
                    query,
                    score.transaction_id,
                    score.model_version,
                    score.risk_score,
                    score.risk_label.value,
                    score.threshold_used,
                    score.was_flagged,
                    score.scoring_latency_ms,
                    score.feature_fetch_latency_ms,
                )
        except Exception as exc:
            logger.debug("Score log persistence skipped (offline/test mode)", error=str(exc))
