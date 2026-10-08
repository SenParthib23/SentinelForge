"""
SentinelForge — tests/integration/test_scoring_pipeline.py

PURPOSE:
    Integration tests verifying end-to-end scoring pipeline:
    TransactionEvent → StreamingFeaturePipeline → Redis Online Store → FraudScorer → ThresholdRouter.
"""

from decimal import Decimal

import pytest

from services.feature_engine.faust_app import StreamingFeaturePipeline
from services.feature_engine.online_store import OnlineFeatureStore
from services.scoring.feature_fetcher import FeatureFetcher
from services.scoring.scorer import FraudScorer
from services.scoring.threshold_router import ThresholdRouter
from shared.models.fraud_case import RiskLabel
from shared.models.transaction import PaymentChannel, TransactionEvent


@pytest.mark.asyncio
async def test_end_to_end_scoring_flow(
    sample_transaction_event: TransactionEvent,
) -> None:
    """Verify end-to-end flow from streaming event to risk scoring decision."""
    # 1. Feature Pipeline & Online Store
    online_store = OnlineFeatureStore()
    pipeline = StreamingFeaturePipeline(online_store=online_store)

    # 2. Process incoming transaction event
    vector = await pipeline.process_transaction(sample_transaction_event)
    assert vector.transaction_id == sample_transaction_event.transaction_id

    # 3. Initialize Scorer with same online store
    fetcher = FeatureFetcher(online_store=online_store)
    scorer = FraudScorer(feature_fetcher=fetcher)

    # 4. Synchronously score the transaction
    score = await scorer.score_transaction(
        transaction_id=sample_transaction_event.transaction_id,
        event=sample_transaction_event,
    )

    assert score.transaction_id == sample_transaction_event.transaction_id
    assert 0.0 <= score.risk_score <= 1.0
    assert score.scoring_latency_ms >= 0.0
    assert score.feature_fetch_latency_ms >= 0.0
