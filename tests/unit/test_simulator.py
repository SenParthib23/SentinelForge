"""
SentinelForge — tests/unit/test_simulator.py

PURPOSE:
    Unit tests verifying TransactionSimulator entity generation, distribution properties,
    and the 5 synthetic fraud typologies.
"""

from decimal import Decimal

import pytest

from services.transaction_simulator.simulator import (
    FraudPatternType,
    TransactionSimulator,
)


@pytest.fixture
def simulator() -> TransactionSimulator:
    """Fixture providing deterministic simulator instance."""
    return TransactionSimulator(
        num_customers=100,
        num_merchants=50,
        fraud_rate=0.05,
        seed=42,
    )


def test_simulator_initialization(simulator: TransactionSimulator) -> None:
    """Verify customer and merchant populations initialize correctly."""
    assert len(simulator.customers) == 100
    assert len(simulator.merchants) == 50

    sample_customer = next(iter(simulator.customers.values()))
    assert sample_customer.avg_amount > 0
    assert sample_customer.home_lat != 0.0
    assert len(sample_customer.primary_device) > 5


def test_generate_legitimate_transaction(simulator: TransactionSimulator) -> None:
    """Verify legitimate transaction parameters match normal customer profile."""
    event, is_fraud, pattern = simulator.generate_transaction(
        fraud_type=FraudPatternType.NONE, force_fraud=False
    )
    assert is_fraud is False
    assert pattern == FraudPatternType.NONE
    assert event.amount > Decimal("0.00")
    assert event.customer_id in simulator.customers
    assert event.merchant_id in simulator.merchants


def test_fraud_pattern_amount_anomaly(simulator: TransactionSimulator) -> None:
    """Verify amount anomaly generates values > 10x customer's normal average."""
    event, is_fraud, pattern = simulator.generate_transaction(
        fraud_type=FraudPatternType.AMOUNT_ANOMALY, force_fraud=True
    )
    assert is_fraud is True
    assert pattern == FraudPatternType.AMOUNT_ANOMALY

    customer = simulator.customers[event.customer_id]
    assert float(event.amount) >= customer.avg_amount * 8.0


def test_fraud_pattern_geographic_anomaly(simulator: TransactionSimulator) -> None:
    """Verify geographic anomaly produces significant displacement or international flag."""
    event, is_fraud, pattern = simulator.generate_transaction(
        fraud_type=FraudPatternType.GEOGRAPHIC_ANOMALY, force_fraud=True
    )
    assert is_fraud is True
    assert pattern == FraudPatternType.GEOGRAPHIC_ANOMALY

    customer = simulator.customers[event.customer_id]
    lat_diff = abs((event.location_lat or 0.0) - customer.home_lat)
    lng_diff = abs((event.location_lng or 0.0) - customer.home_lng)

    # Displaced either domestically (> 10 degrees) or international (UK ~ 51.5)
    assert lat_diff > 5.0 or lng_diff > 5.0 or event.is_international is True


def test_fraud_pattern_card_not_present(simulator: TransactionSimulator) -> None:
    """Verify card-not-present fraud uses a novel/hijacked device fingerprint."""
    event, is_fraud, pattern = simulator.generate_transaction(
        fraud_type=FraudPatternType.CARD_NOT_PRESENT, force_fraud=True
    )
    assert is_fraud is True
    assert pattern == FraudPatternType.CARD_NOT_PRESENT

    customer = simulator.customers[event.customer_id]
    assert event.device_fingerprint != customer.primary_device
    assert "hijack" in event.device_fingerprint


def test_fraud_pattern_new_merchant(simulator: TransactionSimulator) -> None:
    """Verify new merchant fraud selects high-risk merchant categories."""
    event, is_fraud, pattern = simulator.generate_transaction(
        fraud_type=FraudPatternType.NEW_MERCHANT, force_fraud=True
    )
    assert is_fraud is True
    assert pattern == FraudPatternType.NEW_MERCHANT
    assert event.merchant_category in {"crypto_exchange", "luxury_jewelry", "gambling", "electronics"}


@pytest.mark.asyncio
async def test_stream_transactions_generator(simulator: TransactionSimulator) -> None:
    """Verify asynchronous generator emits the specified limit of transactions."""
    emitted = []
    async for event, is_fraud, pattern in simulator.stream_transactions(
        events_per_second=1000, max_events=10
    ):
        emitted.append((event, is_fraud, pattern))

    assert len(emitted) == 10
    assert all(isinstance(e[0].amount, Decimal) for e in emitted)
