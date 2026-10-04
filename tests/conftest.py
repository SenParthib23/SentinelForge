"""
SentinelForge — tests/conftest.py

PURPOSE:
    Provides shared Pytest fixtures, test environment configurations,
    and mock database objects across unit and integration test suites.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from shared.config.settings import Settings
from shared.models.feature_vector import FeatureVector
from shared.models.fraud_case import (
    AnalystDecision,
    AnalystFeedback,
    CaseStatus,
    FraudCase,
    FraudScore,
    RiskLabel,
)
from shared.models.model_version import ModelStatus, ModelVersion
from shared.models.transaction import PaymentChannel, TransactionEvent


@pytest.fixture
def mock_settings() -> Settings:
    """Fixture providing test configuration settings."""
    return Settings(
        ENVIRONMENT="test",
        DEBUG=True,
        POSTGRES_HOST="localhost",
        POSTGRES_PORT=5433,
        POSTGRES_DB="sentinelforge_test",
        POSTGRES_USER="test_user",
        POSTGRES_PASSWORD="test_password",
        REDIS_HOST="localhost",
        REDIS_PORT=6380,
        REDIS_DB=15,
        KAFKA_BOOTSTRAP_SERVERS="localhost:9092",
        GROQ_API_KEY="gsk_test_mock_key",
    )


@pytest.fixture
def sample_transaction_event() -> TransactionEvent:
    """Fixture providing a valid sample TransactionEvent."""
    return TransactionEvent(
        transaction_id="txn_test_1001",
        customer_id="cust_9876",
        merchant_id="merch_5432",
        amount=Decimal("499.50"),
        currency="INR",
        channel=PaymentChannel.UPI,
        device_fingerprint="fp_hash_a1b2c3d4",
        ip_address="192.168.1.50",
        location_lat=19.0760,
        location_lng=72.8777,
        merchant_category="electronics",
        is_international=False,
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_feature_vector(sample_transaction_event: TransactionEvent) -> FeatureVector:
    """Fixture providing a valid sample FeatureVector matching the sample transaction."""
    return FeatureVector(
        transaction_id=sample_transaction_event.transaction_id,
        customer_id=sample_transaction_event.customer_id,
        txn_count_1min=2,
        txn_count_5min=4,
        txn_count_1hr=6,
        txn_count_24hr=12,
        avg_amount_1hr=350.0,
        max_amount_24hr=1500.0,
        amount_zscore=1.25,
        unique_devices_1hr=1,
        device_txn_count=3,
        is_new_device=False,
        merchant_avg_amount=420.0,
        merchant_fraud_rate=0.015,
        is_new_merchant=False,
        distance_from_last_txn=2.4,
        is_international=False,
        hour_of_day=14,
        is_weekend=False,
        feature_version="1.0.0",
    )
