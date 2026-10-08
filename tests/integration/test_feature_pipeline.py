"""
SentinelForge — tests/integration/test_feature_pipeline.py

PURPOSE:
    Integration tests verifying end-to-end streaming feature computation,
    online store Redis retrieval fidelity, and offline batch transformation consistency.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from services.feature_engine.faust_app import StreamingFeaturePipeline
from services.feature_engine.offline_store import OfflineFeatureStore
from services.feature_engine.online_store import OnlineFeatureStore
from shared.models.transaction import PaymentChannel, TransactionEvent


@pytest.mark.asyncio
async def test_end_to_end_streaming_pipeline(
    sample_transaction_event: TransactionEvent,
) -> None:
    """Verify StreamingFeaturePipeline computes, validates, and stores FeatureVector."""
    online_store = OnlineFeatureStore()
    pipeline = StreamingFeaturePipeline(online_store=online_store)

    # Process first transaction
    vector = await pipeline.process_transaction(sample_transaction_event)

    assert vector.transaction_id == sample_transaction_event.transaction_id
    assert vector.customer_id == sample_transaction_event.customer_id
    assert vector.txn_count_1min == 1
    assert vector.avg_amount_1hr == float(sample_transaction_event.amount)
    assert vector.is_new_device is True
    assert vector.is_new_merchant is True

    # Retrieve from online store
    retrieved = await online_store.get_feature_vector(sample_transaction_event.transaction_id)
    assert retrieved is not None
    assert retrieved.transaction_id == sample_transaction_event.transaction_id
    assert retrieved.avg_amount_1hr == vector.avg_amount_1hr


@pytest.mark.asyncio
async def test_offline_batch_feature_extraction() -> None:
    """Verify OfflineFeatureStore applies identical feature formulas across batch records."""
    offline_store = OfflineFeatureStore()

    records = [
        {
            "transaction_id": "txn_b1",
            "customer_id": "cust_100",
            "merchant_id": "merch_10",
            "amount": 250.0,
            "currency": "INR",
            "channel": "upi",
            "device_fingerprint": "dev_alpha",
            "location_lat": 19.076,
            "location_lng": 72.877,
            "merchant_category": "grocery",
            "is_international": False,
            "timestamp": "2026-10-01T10:00:00Z",
            "is_fraud": 0,
        },
        {
            "transaction_id": "txn_b2",
            "customer_id": "cust_100",
            "merchant_id": "merch_10",
            "amount": 500.0,
            "currency": "INR",
            "channel": "upi",
            "device_fingerprint": "dev_alpha",
            "location_lat": 19.078,
            "location_lng": 72.879,
            "merchant_category": "grocery",
            "is_international": False,
            "timestamp": "2026-10-01T10:02:00Z",
            "is_fraud": 0,
        },
    ]
    df = pd.DataFrame(records)
    features_df = offline_store.extract_batch_features(df)

    assert len(features_df) == 2
    assert "txn_count_5min" in features_df.columns
    assert "amount_zscore" in features_df.columns
    assert "distance_from_last_txn" in features_df.columns

    # Second transaction within 2 minutes: velocity should be 2
    assert features_df.iloc[1]["txn_count_5min"] == 2
    assert features_df.iloc[1]["is_new_device"] == 0  # Re-used device
    assert features_df.iloc[1]["is_new_merchant"] == 0  # Re-used merchant
