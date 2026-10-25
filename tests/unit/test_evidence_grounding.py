"""
SentinelForge — tests/unit/test_evidence_grounding.py

PURPOSE:
    Unit tests verifying EvidenceGroundingVerifier citation extraction,
    numerical proximity checks, and hallucination detection.
"""

import pytest

from services.investigation.evidence_grounding import EvidenceGroundingVerifier


@pytest.fixture
def sample_context_bundle() -> dict:
    """Fixture providing verified sample context bundle."""
    return {
        "transaction_info": {
            "transaction_id": "txn_test_1001",
            "amount": 25000.0,
            "channel": "card",
        },
        "scoring_info": {
            "risk_score": 0.92,
            "risk_label": "critical",
        },
        "customer_profile": {
            "avg_amount": 1200.0,
        },
        "feature_snapshot": {
            "amount_zscore": 19.8,
            "txn_count_1min": 5,
            "is_new_device": True,
            "distance_from_last_txn": 7180.5,
        },
        "linked_accounts_graph": {
            "shared_device_or_ip_count": 3,
        },
    }


def test_grounded_narrative_passes(sample_context_bundle: dict) -> None:
    """Verify narrative citing real context facts passes verification."""
    verifier = EvidenceGroundingVerifier(minimum_grounding_score=0.80)

    narrative = """
    ## Transaction Summary
    Transaction of INR 25,000 flagged for customer.

    ## Risk Indicators
    - [STRONG INDICATOR] Extreme Amount Anomaly [Citing: amount_zscore=19.8, amount=25000].
    - [STRONG INDICATOR] Mule Hardware Sharing [Citing: shared_device_or_ip_count=3].
    - [STRONG INDICATOR] Impossible Travel [Citing: distance_from_last_txn=7180.5].
    - [MODERATE INDICATOR] High Velocity [Citing: txn_count_1min=5].
    - [MODERATE INDICATOR] Novel Device [Citing: is_new_device=true].

    ## Recommended Action
    ESCALATE_URGENT
    """

    score, verified, ungrounded, is_acceptable = verifier.verify_narrative(
        narrative, sample_context_bundle
    )

    assert is_acceptable is True
    assert score >= 0.80
    assert len(verified) >= 5
    assert len(ungrounded) == 0


def test_ungrounded_claims_detected(sample_context_bundle: dict) -> None:
    """Verify fabricated/hallucinated features drop grounding score and fail acceptance."""
    verifier = EvidenceGroundingVerifier(minimum_grounding_score=0.80)

    hallucinated_narrative = """
    ## Risk Indicators
    - [STRONG INDICATOR] Fabricated feature [Citing: fake_feature_z=999.0].
    - [STRONG INDICATOR] Wrong amount [Citing: amount=999999.0].
    - [STRONG INDICATOR] Non-existent linked accounts [Citing: shared_device_or_ip_count=50].
    - [MODERATE INDICATOR] Valid claim [Citing: txn_count_1min=5].
    """

    score, verified, ungrounded, is_acceptable = verifier.verify_narrative(
        hallucinated_narrative, sample_context_bundle
    )

    assert is_acceptable is False
    assert score < 0.80
    assert len(ungrounded) >= 3


def test_missing_citations_flagged(sample_context_bundle: dict) -> None:
    """Verify narrative lacking formal bracketed citations is flagged."""
    verifier = EvidenceGroundingVerifier(minimum_grounding_score=0.80)

    unreferenced_narrative = """
    The transaction looks very suspicious because the customer spent a lot of money and
    it came from a foreign country. Please escalate.
    """

    score, verified, ungrounded, is_acceptable = verifier.verify_narrative(
        unreferenced_narrative, sample_context_bundle
    )

    assert is_acceptable is False
    assert len(ungrounded) > 0
