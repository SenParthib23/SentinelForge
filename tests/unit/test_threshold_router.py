"""
SentinelForge — tests/unit/test_threshold_router.py

PURPOSE:
    Unit tests verifying 4-tier operational risk routing and alert dispatch logic.
"""

import pytest

from services.scoring.threshold_router import ThresholdRouter
from shared.models.fraud_case import RiskLabel


def test_threshold_router_low_risk() -> None:
    """Verify low risk scores are categorized as LOW and not flagged."""
    router = ThresholdRouter(low_threshold=0.30, high_threshold=0.60, critical_threshold=0.85)

    label, flagged, threshold = router.evaluate_risk(0.15)
    assert label == RiskLabel.LOW
    assert flagged is False
    assert threshold == 0.30


def test_threshold_router_medium_risk() -> None:
    """Verify medium risk scores are categorized as MEDIUM and not flagged."""
    router = ThresholdRouter(low_threshold=0.30, high_threshold=0.60, critical_threshold=0.85)

    label, flagged, threshold = router.evaluate_risk(0.45)
    assert label == RiskLabel.MEDIUM
    assert flagged is False
    assert threshold == 0.60


def test_threshold_router_high_risk_flagged() -> None:
    """Verify high risk scores are categorized as HIGH and flagged for review."""
    router = ThresholdRouter(low_threshold=0.30, high_threshold=0.60, critical_threshold=0.85)

    label, flagged, threshold = router.evaluate_risk(0.72)
    assert label == RiskLabel.HIGH
    assert flagged is True
    assert threshold == 0.60


def test_threshold_router_critical_risk_urgent() -> None:
    """Verify critical risk scores are categorized as CRITICAL and flagged as urgent."""
    router = ThresholdRouter(low_threshold=0.30, high_threshold=0.60, critical_threshold=0.85)

    label, flagged, threshold = router.evaluate_risk(0.95)
    assert label == RiskLabel.CRITICAL
    assert flagged is True
    assert threshold == 0.85
