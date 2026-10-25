"""
SentinelForge — tests/unit/test_case_builder.py

PURPOSE:
    Unit tests verifying CaseBuilder context enrichment, linked account detection,
    and fraud case persistence.
"""

from decimal import Decimal

import pytest

from services.feature_engine.graph_features import GraphFeatureComputer
from services.investigation.case_builder import CaseBuilder
from shared.models.feature_vector import FeatureVector
from shared.models.fraud_case import CaseStatus, RiskLabel
from shared.models.transaction import PaymentChannel, TransactionEvent


@pytest.mark.asyncio
async def test_case_builder_context_assembly(
    sample_transaction_event: TransactionEvent,
    sample_feature_vector: FeatureVector,
) -> None:
    """Verify CaseBuilder compiles all 7 context bundle dimensions."""
    graph = GraphFeatureComputer()
    # Add linked accounts to graph
    graph.record_interaction("cust_9876", "fp_hash_a1b2c3d4", "192.168.1.50")
    graph.record_interaction("cust_mule_1", "fp_hash_a1b2c3d4", "192.168.1.50")

    builder = CaseBuilder(graph_computer=graph)
    builder.register_customer_profile("cust_9876", avg_amount=350.0, std_amount=50.0, home_city="Bengaluru")

    bundle = await builder.build_case_context(
        transaction_id=sample_transaction_event.transaction_id,
        risk_score=0.88,
        risk_label=RiskLabel.CRITICAL,
        event=sample_transaction_event,
        feature_vector=sample_feature_vector,
    )

    # 1. Transaction info
    assert bundle["transaction_info"]["transaction_id"] == sample_transaction_event.transaction_id
    assert bundle["transaction_info"]["amount"] == float(sample_transaction_event.amount)

    # 2. Scoring info
    assert bundle["scoring_info"]["risk_score"] == 0.88
    assert bundle["scoring_info"]["risk_label"] == "critical"

    # 3. Customer profile
    assert bundle["customer_profile"]["home_city"] == "Bengaluru"
    assert bundle["customer_profile"]["avg_amount"] == 350.0

    # 4. Linked accounts graph
    assert bundle["linked_accounts_graph"]["shared_device_or_ip_count"] == 1
    assert "cust_mule_1" in bundle["linked_accounts_graph"]["linked_customer_ids"]

    # 5. Merchant profile
    assert bundle["merchant_profile"]["category"] == "electronics"

    # 6. Geographic signals
    assert bundle["geographic_signals"]["distance_from_last_txn_km"] == sample_feature_vector.distance_from_last_txn

    # 7. Feature snapshot
    assert "amount_zscore" in bundle["feature_snapshot"]


@pytest.mark.asyncio
async def test_case_builder_persistence_and_update() -> None:
    """Verify case creation and narrative update in cache."""
    builder = CaseBuilder()
    context = {"test_key": "test_val"}

    case = await builder.create_and_persist_case("txn_c1", 0.75, context)
    assert case.transaction_id == "txn_c1"
    assert case.status == CaseStatus.PENDING

    # Update narrative
    await builder.update_case_narrative("txn_c1", "## Transaction Summary\nSuspicious spike.")
    updated = builder.get_cached_case("txn_c1")
    assert updated is not None
    assert updated.status == CaseStatus.INVESTIGATING
    assert "Suspicious spike" in updated.investigation_narrative
