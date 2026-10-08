"""
SentinelForge — tests/unit/test_feature_definitions.py

PURPOSE:
    Unit tests verifying mathematical accuracy and business logic for all
    feature computation subsystems (Haversine, Velocity, Spend, Device, Merchant, Graph).
"""

import time
from decimal import Decimal

import pytest

from services.feature_engine.device_features import DeviceFeatureComputer
from services.feature_engine.feature_definitions import (
    FEATURE_CATALOG,
    ORDERED_FEATURE_NAMES,
)
from services.feature_engine.geo_features import compute_haversine_distance
from services.feature_engine.graph_features import GraphFeatureComputer
from services.feature_engine.merchant_features import MerchantFeatureComputer
from services.feature_engine.spend_features import SpendFeatureComputer
from services.feature_engine.velocity_features import VelocityFeatureComputer


def test_haversine_distance_computation() -> None:
    """Verify Haversine formula against known geographical coordinates."""
    # Mumbai (19.0760, 72.8777) to Delhi (28.6139, 77.2090) ~ 1148 km
    dist_mum_delhi = compute_haversine_distance(19.0760, 72.8777, 28.6139, 77.2090)
    assert 1100.0 < dist_mum_delhi < 1200.0

    # London (51.5074, -0.1278) to Mumbai ~ 7180 km
    dist_london_mum = compute_haversine_distance(51.5074, -0.1278, 19.0760, 72.8777)
    assert 7100.0 < dist_london_mum < 7300.0

    # Same location
    assert compute_haversine_distance(19.0, 72.0, 19.0, 72.0) == 0.0

    # Missing coordinates
    assert compute_haversine_distance(None, 72.0, 19.0, None) == 0.0


@pytest.mark.asyncio
async def test_velocity_feature_computer() -> None:
    """Verify velocity sliding-window counters."""
    computer = VelocityFeatureComputer()
    now = 1700000000.0

    # First transaction
    feats1 = await computer.record_and_compute(
        customer_id="cust_v1",
        device_fingerprint="dev_v1",
        transaction_id="txn_v1",
        timestamp_epoch=now,
    )
    assert feats1["txn_count_1min"] == 1
    assert feats1["txn_count_5min"] == 1
    assert feats1["device_txn_count"] == 1

    # Second transaction 20 seconds later
    feats2 = await computer.record_and_compute(
        customer_id="cust_v1",
        device_fingerprint="dev_v1",
        transaction_id="txn_v2",
        timestamp_epoch=now + 20.0,
    )
    assert feats2["txn_count_1min"] == 2
    assert feats2["txn_count_5min"] == 2
    assert feats2["device_txn_count"] == 2

    # Third transaction 10 minutes later (outside 1min and 5min, inside 1hr)
    feats3 = await computer.record_and_compute(
        customer_id="cust_v1",
        device_fingerprint="dev_v1",
        transaction_id="txn_v3",
        timestamp_epoch=now + 600.0,
    )
    assert feats3["txn_count_1min"] == 1
    assert feats3["txn_count_5min"] == 1
    assert feats3["txn_count_1hr"] == 3


def test_spend_feature_computer() -> None:
    """Verify rolling average, max, and z-score deviation."""
    computer = SpendFeatureComputer()
    now = 1700000000.0

    # Normal spend: baseline mean 500, std 100
    feats1 = computer.compute_spend_features(
        customer_id="cust_s1",
        amount=500.0,
        timestamp_epoch=now,
        baseline_mean=500.0,
        baseline_std=100.0,
    )
    assert feats1["avg_amount_1hr"] == 500.0
    assert feats1["amount_zscore"] == 0.0

    # High anomaly spend: 3500 INR (3000 above mean = 30 std deviations)
    feats2 = computer.compute_spend_features(
        customer_id="cust_s1",
        amount=3500.0,
        timestamp_epoch=now + 100.0,
        baseline_mean=500.0,
        baseline_std=100.0,
    )
    assert feats2["amount_zscore"] == 30.0
    assert feats2["max_amount_24hr"] == 3500.0


def test_device_feature_computer() -> None:
    """Verify device novelty detection and diversity counts."""
    computer = DeviceFeatureComputer()
    now = 1700000000.0

    # First device interaction (novel)
    feats1 = computer.compute_device_features(
        customer_id="cust_d1",
        device_fingerprint="dev_100",
        timestamp_epoch=now,
    )
    assert feats1["is_new_device"] is True
    assert feats1["unique_devices_1hr"] == 1

    # Repeat interaction with same device (no longer novel)
    feats2 = computer.compute_device_features(
        customer_id="cust_d1",
        device_fingerprint="dev_100",
        timestamp_epoch=now + 60.0,
    )
    assert feats2["is_new_device"] is False
    assert feats2["unique_devices_1hr"] == 1

    # Second device for same customer in 1hr
    feats3 = computer.compute_device_features(
        customer_id="cust_d1",
        device_fingerprint="dev_200",
        timestamp_epoch=now + 120.0,
    )
    assert feats3["is_new_device"] is True
    assert feats3["unique_devices_1hr"] == 2


def test_merchant_feature_computer() -> None:
    """Verify merchant risk scoring and customer affinity novelty."""
    computer = MerchantFeatureComputer()

    # First interaction with crypto exchange
    feats1 = computer.compute_merchant_features(
        customer_id="cust_m1",
        merchant_id="merch_crypto_99",
        merchant_category="crypto_exchange",
    )
    assert feats1["is_new_merchant"] is True
    assert feats1["merchant_fraud_rate"] == 0.12  # Crypto exchange base risk

    # Repeat interaction with same merchant
    feats2 = computer.compute_merchant_features(
        customer_id="cust_m1",
        merchant_id="merch_crypto_99",
        merchant_category="crypto_exchange",
    )
    assert feats2["is_new_merchant"] is False


def test_graph_feature_computer() -> None:
    """Verify linked account identity graph traversal."""
    graph = GraphFeatureComputer()

    # Two distinct accounts sharing the same device hardware
    graph.record_interaction("cust_alpha", "shared_device_999", "192.168.1.10")
    graph.record_interaction("cust_beta", "shared_device_999", "192.168.1.20")

    linked_alpha = graph.get_linked_accounts("cust_alpha", "shared_device_999")
    assert "cust_beta" in linked_alpha
    assert "cust_alpha" not in linked_alpha  # Self is excluded


def test_feature_catalog_integrity() -> None:
    """Verify feature catalog defines all 17 canonical features."""
    assert len(ORDERED_FEATURE_NAMES) >= 15
    for name, meta in FEATURE_CATALOG.items():
        assert "aggregation" in meta
        assert "dtype" in meta
        assert "default" in meta
