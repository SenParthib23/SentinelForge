"""
SentinelForge — tests/unit/test_canary_router.py

PURPOSE:
    Unit tests for CanaryRouter sticky customer hashing, traffic split ratios, and shadow scoring.
"""

import pytest

from services.scoring.canary_router import CanaryConfig, CanaryRouter


def test_canary_default_routing_to_champion() -> None:
    """Verify default routing directs 100% of traffic to champion."""
    router = CanaryRouter()
    model, is_canary = router.route_transaction("cust_1001")
    assert model == "baseline_v1"
    assert is_canary is False


def test_sticky_customer_routing_consistency() -> None:
    """Verify identical customer_id always routes to the same model version."""
    router = CanaryRouter(
        CanaryConfig(
            champion_version="baseline_v1",
            canary_version="candidate_v2",
            canary_percentage=25.0,
            is_active=True,
        )
    )

    cust_a = "cust_alpha_123"
    model_1, is_canary_1 = router.route_transaction(cust_a)
    model_2, is_canary_2 = router.route_transaction(cust_a)
    model_3, is_canary_3 = router.route_transaction(cust_a)

    assert model_1 == model_2 == model_3
    assert is_canary_1 == is_canary_2 == is_canary_3


def test_traffic_split_proportions() -> None:
    """Verify split distributes roughly according to target percentage."""
    router = CanaryRouter()
    router.update_split(canary_percentage=30.0, canary_version="candidate_v2")

    n_samples = 1000
    canary_count = 0
    for i in range(n_samples):
        _, is_canary = router.route_transaction(f"customer_random_{i}")
        if is_canary:
            canary_count += 1

    observed_pct = (canary_count / n_samples) * 100.0
    # Allow reasonable binomial distribution tolerance (+/- 5%)
    assert 25.0 <= observed_pct <= 35.0


def test_shadow_mode_execution() -> None:
    """Verify in shadow mode primary traffic stays on champion while shadow flag is True."""
    router = CanaryRouter()
    router.update_split(canary_percentage=0.0, canary_version="candidate_v2", shadow_mode=True)

    model, is_canary = router.route_transaction("cust_test")
    assert model == "baseline_v1"
    assert is_canary is False
    assert router.should_shadow_score() is True


def test_reset_to_champion() -> None:
    """Verify reset_to_champion immediately reverts all traffic."""
    router = CanaryRouter()
    router.update_split(canary_percentage=50.0, canary_version="candidate_v2")
    assert router.config.canary_percentage == 50.0

    router.reset_to_champion()
    assert router.config.canary_percentage == 0.0
    assert router.config.is_active is False
