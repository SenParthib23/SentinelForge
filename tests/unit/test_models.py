"""
SentinelForge — tests/unit/test_models.py

PURPOSE:
    Unit tests verifying Pydantic schema validation, boundaries, serialization,
    and business rule enforcement across core data models.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from shared.models.feature_vector import FeatureVector
from shared.models.fraud_case import (
    AnalystDecision,
    AnalystFeedback,
    CaseStatus,
    FraudCase,
    FraudScore,
    InvestigationNarrative,
    RiskLabel,
)
from shared.models.model_version import (
    CanaryMetrics,
    EvalReport,
    ModelStatus,
    ModelVersion,
)
from shared.models.transaction import PaymentChannel, TransactionEvent


def test_transaction_event_valid(sample_transaction_event: TransactionEvent) -> None:
    """Verify valid TransactionEvent serializes and populates fields correctly."""
    assert sample_transaction_event.transaction_id == "txn_test_1001"
    assert sample_transaction_event.amount == Decimal("499.50")
    assert sample_transaction_event.currency == "INR"
    assert sample_transaction_event.channel == PaymentChannel.UPI


def test_transaction_event_invalid_amount() -> None:
    """Verify non-positive amounts trigger validation error."""
    with pytest.raises(ValidationError):
        TransactionEvent(
            customer_id="c1",
            merchant_id="m1",
            amount=Decimal("-10.00"),  # Negative amount rejected
            device_fingerprint="fp1",
        )


def test_transaction_event_currency_normalization() -> None:
    """Verify currency string is coerced to uppercase."""
    txn = TransactionEvent(
        customer_id="c1",
        merchant_id="m1",
        amount=Decimal("100.00"),
        currency="inr ",
        device_fingerprint="fp1",
    )
    assert txn.currency == "INR"


def test_transaction_event_coordinate_validation() -> None:
    """Verify out-of-bounds latitude/longitude trigger validation error."""
    with pytest.raises(ValidationError):
        TransactionEvent(
            customer_id="c1",
            merchant_id="m1",
            amount=Decimal("100.00"),
            device_fingerprint="fp1",
            location_lat=120.0,  # Latitude > 90.0 invalid
        )


def test_feature_vector_to_model_input_dict(
    sample_feature_vector: FeatureVector,
) -> None:
    """
    Verify to_model_input_dict converts feature vector into pure numeric
    dictionary suitable for tabular ML models like LightGBM.
    """
    model_inputs = sample_feature_vector.to_model_input_dict()

    # Meta-identifiers must be excluded
    assert "transaction_id" not in model_inputs
    assert "customer_id" not in model_inputs
    assert "computed_at" not in model_inputs
    assert "feature_version" not in model_inputs

    # Numerical features must be present
    assert model_inputs["txn_count_1min"] == 2.0 or model_inputs["txn_count_1min"] == 2
    assert model_inputs["amount_zscore"] == 1.25
    assert model_inputs["is_new_device"] == 0.0 or model_inputs["is_new_device"] == 0
    assert model_inputs["is_weekend"] == 0.0 or model_inputs["is_weekend"] == 0


def test_fraud_score_validation() -> None:
    """Verify FraudScore boundaries and risk labels."""
    score = FraudScore(
        transaction_id="txn_001",
        model_version="v1",
        risk_score=0.87,
        risk_label=RiskLabel.CRITICAL,
        threshold_used=0.85,
        was_flagged=True,
        scoring_latency_ms=4.2,
        feature_fetch_latency_ms=0.8,
    )
    assert score.risk_score == 0.87
    assert score.was_flagged is True
    assert score.risk_label == RiskLabel.CRITICAL

    # Invalid probability (>1.0)
    with pytest.raises(ValidationError):
        FraudScore(
            transaction_id="txn_001",
            model_version="v1",
            risk_score=1.5,
            risk_label=RiskLabel.HIGH,
            threshold_used=0.6,
            was_flagged=True,
            scoring_latency_ms=4.0,
            feature_fetch_latency_ms=1.0,
        )


def test_investigation_narrative_and_case() -> None:
    """Verify InvestigationNarrative and FraudCase lifecycle fields."""
    narrative = InvestigationNarrative(
        transaction_summary="Transaction of 50,000 INR from new device",
        risk_indicators=[
            {"indicator": "New Device Fingerprint", "tag": "[STRONG INDICATOR]", "citation": "device_fingerprint"}
        ],
        customer_context="Customer average spend is 1,200 INR",
        recommended_action="HOLD_FOR_REVIEW",
        evidence_grounding_score=0.95,
    )

    case = FraudCase(
        transaction_id="txn_100",
        risk_score=0.78,
        case_context={"customer_id": "c1", "avg_spend": 1200},
        investigation_narrative=narrative.transaction_summary,
        status=CaseStatus.PENDING,
    )
    assert case.status == CaseStatus.PENDING
    assert case.risk_score == 0.78


def test_analyst_feedback() -> None:
    """Verify AnalystFeedback schema captures human decision correctly."""
    feedback = AnalystFeedback(
        analyst_id="analyst_jane",
        decision=AnalystDecision.REJECTED,
        notes="Confirmed account takeover via phone phishing report",
    )
    assert feedback.decision == AnalystDecision.REJECTED
    assert feedback.analyst_id == "analyst_jane"


def test_model_version_and_eval_report() -> None:
    """Verify ModelVersion and EvalReport lifecycle structures."""
    eval_report = EvalReport(
        model_version="v2",
        baseline_version="v1",
        auc_roc=0.962,
        precision=0.915,
        recall=0.890,
        f1_score=0.902,
        precision_at_90_recall=0.885,
        p99_latency_ms=6.8,
        passed_benchmark=True,
        passed_regression=True,
    )
    assert eval_report.passed_benchmark is True
    assert eval_report.passed_regression is True

    model_ver = ModelVersion(
        model_name="lightgbm_fraud",
        version="v2",
        status=ModelStatus.CANARY,
        eval_metrics=eval_report.model_dump(),
    )
    assert model_ver.status == ModelStatus.CANARY
    assert model_ver.version == "v2"
