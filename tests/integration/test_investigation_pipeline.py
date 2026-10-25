"""
SentinelForge — tests/integration/test_investigation_pipeline.py

PURPOSE:
    Integration tests verifying end-to-end asynchronous investigation pipeline:
    Flagged TransactionEvent → CaseBuilder (360 Enrichment) → InvestigationAgent
    (Groq / Factual Synthesis) → Evidence Grounding Verification → FraudCase Persistence.
"""

from decimal import Decimal

import pytest

from services.feature_engine.graph_features import GraphFeatureComputer
from services.investigation.case_builder import CaseBuilder
from services.investigation.evidence_grounding import EvidenceGroundingVerifier
from services.investigation.investigation_agent import InvestigationAgent
from shared.models.feature_vector import FeatureVector
from shared.models.fraud_case import CaseStatus, InvestigationNarrative, RiskLabel
from shared.models.transaction import PaymentChannel, TransactionEvent


@pytest.mark.asyncio
async def test_end_to_end_investigation_pipeline(
    sample_transaction_event: TransactionEvent,
    sample_feature_vector: FeatureVector,
) -> None:
    """Verify complete asynchronous case enrichment and narrative generation flow."""
    # 1. Initialize pipeline components
    graph = GraphFeatureComputer()
    graph.record_interaction("cust_9876", "fp_hash_a1b2c3d4")
    graph.record_interaction("cust_mule_88", "fp_hash_a1b2c3d4")

    case_builder = CaseBuilder(graph_computer=graph)
    verifier = EvidenceGroundingVerifier(minimum_grounding_score=0.80)
    agent = InvestigationAgent(verifier=verifier)

    # 2. Assemble 360-degree context bundle
    context = await case_builder.build_case_context(
        transaction_id=sample_transaction_event.transaction_id,
        risk_score=0.89,
        risk_label=RiskLabel.CRITICAL,
        event=sample_transaction_event,
        feature_vector=sample_feature_vector,
    )

    assert "transaction_info" in context
    assert "linked_accounts_graph" in context
    assert context["linked_accounts_graph"]["shared_device_or_ip_count"] == 1

    # 3. Create initial case record in database / cache
    case = await case_builder.create_and_persist_case(
        transaction_id=sample_transaction_event.transaction_id,
        risk_score=0.89,
        context_bundle=context,
    )
    assert case.status == CaseStatus.PENDING

    # 4. Generate grounded investigation narrative via Agent
    narrative = await agent.generate_narrative(context)

    assert isinstance(narrative, InvestigationNarrative)
    assert narrative.evidence_grounding_score >= 0.80
    assert len(narrative.risk_indicators) > 0
    assert narrative.recommended_action in {"APPROVE", "HOLD_FOR_REVIEW", "ESCALATE_URGENT"}

    # 5. Update case record with completed narrative
    await case_builder.update_case_narrative(
        transaction_id=sample_transaction_event.transaction_id,
        narrative=narrative.raw_markdown or narrative.transaction_summary,
    )

    updated_case = case_builder.get_cached_case(sample_transaction_event.transaction_id)
    assert updated_case is not None
    assert updated_case.status == CaseStatus.INVESTIGATING
    assert updated_case.investigation_narrative is not None
