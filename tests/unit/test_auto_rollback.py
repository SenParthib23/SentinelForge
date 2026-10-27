"""
SentinelForge — tests/unit/test_auto_rollback.py

PURPOSE:
    Unit tests for AutoRollbackManager circuit breaker tracking canary SLAs and triggering rollbacks.
"""

import pytest

from services.dashboard.sse_streamer import CaseStreamer
from services.scoring.auto_rollback import AutoRollbackManager, RollbackThresholds
from services.scoring.canary_router import CanaryConfig, CanaryRouter


@pytest.mark.asyncio
async def test_canary_healthy_no_rollback() -> None:
    """Verify healthy canary execution within SLAs does not trip circuit breaker."""
    router = CanaryRouter(
        CanaryConfig(champion_version="baseline_v1", canary_version="cand_v2", canary_percentage=20.0, is_active=True)
    )
    mgr = AutoRollbackManager(canary_router=router)

    # Record 30 normal transactions (latency ~4.5ms, 0 errors, 1 flag)
    for _ in range(30):
        mgr.record_canary_metric(latency_ms=4.5, is_error=False, was_flagged=False)
        mgr.record_champion_metric(latency_ms=4.3, is_error=False, was_flagged=False)

    record = await mgr.check_health_and_maybe_rollback()
    assert record is None
    assert router.config.is_active is True


@pytest.mark.asyncio
async def test_canary_latency_spike_triggers_rollback() -> None:
    """Verify p99 latency > 10ms triggers atomic rollback to champion."""
    router = CanaryRouter(
        CanaryConfig(champion_version="baseline_v1", canary_version="cand_v2", canary_percentage=20.0, is_active=True)
    )
    mgr = AutoRollbackManager(
        thresholds=RollbackThresholds(max_latency_p99_ms=10.0, min_eval_samples=10),
        canary_router=router,
    )

    # Record 15 transactions with latency spikes (18ms)
    for _ in range(15):
        mgr.record_canary_metric(latency_ms=18.0, is_error=False, was_flagged=False)

    record = await mgr.check_health_and_maybe_rollback()
    assert record is not None
    assert "latency" in record.reason.lower()
    # Verify traffic was reset
    assert router.config.canary_percentage == 0.0
    assert router.config.is_active is False


@pytest.mark.asyncio
async def test_canary_error_spike_triggers_rollback() -> None:
    """Verify error rate > 0.5% triggers automated rollback."""
    router = CanaryRouter(
        CanaryConfig(champion_version="baseline_v1", canary_version="cand_v2", canary_percentage=20.0, is_active=True)
    )
    mgr = AutoRollbackManager(
        thresholds=RollbackThresholds(max_error_rate=0.01, min_eval_samples=10),
        canary_router=router,
    )

    # Record 10 transactions with errors
    for _ in range(10):
        mgr.record_canary_metric(latency_ms=4.5, is_error=True, was_flagged=False)

    record = await mgr.check_health_and_maybe_rollback()
    assert record is not None
    assert "error rate" in record.reason.lower()
    assert router.config.is_active is False
