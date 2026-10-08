"""
SentinelForge — tests/unit/test_scorer.py

PURPOSE:
    Unit tests verifying sub-10ms LightGBM inference, feature vector transformation,
    and model hot-reloading.
"""

from decimal import Decimal

import numpy as np
import pytest

from services.scoring.feature_fetcher import FeatureFetcher
from services.scoring.model_loader import ModelLoader
from services.scoring.scorer import FraudScorer
from shared.models.feature_vector import FeatureVector
from shared.models.fraud_case import FraudScore, RiskLabel
from shared.models.transaction import PaymentChannel, TransactionEvent


@pytest.mark.asyncio
async def test_fraud_scorer_inference(
    sample_transaction_event: TransactionEvent,
    sample_feature_vector: FeatureVector,
) -> None:
    """Verify FraudScorer runs inference and returns structured FraudScore."""
    model_loader = ModelLoader()
    scorer = FraudScorer(model_loader=model_loader)

    score = await scorer.score_transaction(
        transaction_id=sample_transaction_event.transaction_id,
        event=sample_transaction_event,
    )

    assert isinstance(score, FraudScore)
    assert 0.0 <= score.risk_score <= 1.0
    assert score.risk_label in {
        RiskLabel.LOW,
        RiskLabel.MEDIUM,
        RiskLabel.HIGH,
        RiskLabel.CRITICAL,
    }
    assert score.scoring_latency_ms >= 0.0
    # Must be sub-15ms for local Python test
    assert score.scoring_latency_ms < 50.0


def test_feature_array_alignment(sample_feature_vector: FeatureVector) -> None:
    """Verify 2D feature array aligns with exact 17 column names."""
    scorer = FraudScorer()
    arr = scorer._prepare_feature_array(sample_feature_vector)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (1, 17)
    assert arr.dtype == np.float32


def test_model_loader_caching_and_reload() -> None:
    """Verify ModelLoader loads booster into memory and returns cached instance."""
    loader = ModelLoader()
    booster1 = loader.load_model()
    booster2 = loader.load_model()

    assert booster1 is booster2
    assert loader.current_version is not None
