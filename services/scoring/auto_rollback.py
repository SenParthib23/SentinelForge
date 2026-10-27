"""
SentinelForge — services/scoring/auto_rollback.py

PURPOSE:
    Automated health monitoring and circuit breaker for Canary Deployments.
    Continuously tracks p99 latency, error rates, and anomaly flag rates.
    If canary violates operational SLAs, executes an instant atomic rollback to Champion.

ARCHITECTURE POSITION:
    Layer 10 (Model Deployment & Serving) → Automated Rollback Circuit Breaker.

INTERVIEW TALKING POINT:
    "Manual rollback is too slow when high-velocity payment rails are processing hundreds of
    transactions per second. AutoRollbackManager monitors a sliding window of canary traffic.
    If the canary model breaches our 10ms p99 latency SLA, exceeds 0.5% errors, or produces a
    3x false-positive flag rate spike, the system instantly snaps traffic back to 100% Champion
    in sub-millisecond time and broadcasts an alert to the analyst cockpit."
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from services.dashboard.sse_streamer import CaseStreamer, get_case_streamer
from services.scoring.canary_router import CanaryRouter, get_canary_router
from shared.logging.logger import get_logger

logger = get_logger("scoring.auto_rollback")


@dataclass
class RollbackThresholds:
    """Safety thresholds governing automated canary rollback."""
    max_latency_p99_ms: float = 10.0           # Strict sub-10ms LightGBM SLA
    max_error_rate: float = 0.005              # Max 0.5% execution errors
    max_flag_rate_multiplier: float = 3.0      # Flag rate cannot exceed 3x champion baseline
    min_eval_samples: int = 20                 # Minimum canary samples before triggering


@dataclass
class RollbackAuditRecord:
    """Audit log of an automated or manual rollback event."""
    timestamp: str
    canary_version: str
    champion_version: str
    reason: str
    metrics_at_trigger: Dict[str, Any]


class AutoRollbackManager:
    """
    Monitors canary execution health and trips the circuit breaker upon SLA violation.
    """

    def __init__(
        self,
        thresholds: Optional[RollbackThresholds] = None,
        canary_router: Optional[CanaryRouter] = None,
        streamer: Optional[CaseStreamer] = None,
        window_size: int = 200,
    ) -> None:
        """
        Initialize AutoRollbackManager.

        Args:
            thresholds: Circuit breaker trigger thresholds.
            canary_router: Router governing active traffic distribution.
            streamer: SSE manager broadcasting alerts to dashboard.
            window_size: Moving window of transactions tracked.
        """
        self.thresholds = thresholds or RollbackThresholds()
        self.router = canary_router or get_canary_router()
        self.streamer = streamer or get_case_streamer()

        # Sliding window history: (latency_ms, is_error, was_flagged)
        self._canary_history: Deque[tuple[float, bool, bool]] = deque(maxlen=window_size)
        self._champion_history: Deque[tuple[float, bool, bool]] = deque(maxlen=window_size)

        self.rollback_history: List[RollbackAuditRecord] = []

    def record_canary_metric(self, latency_ms: float, is_error: bool = False, was_flagged: bool = False) -> None:
        """Record canary execution outcome into sliding window."""
        self._canary_history.append((latency_ms, is_error, was_flagged))

    def record_champion_metric(self, latency_ms: float, is_error: bool = False, was_flagged: bool = False) -> None:
        """Record champion execution outcome into sliding window."""
        self._champion_history.append((latency_ms, is_error, was_flagged))

    async def check_health_and_maybe_rollback(self) -> Optional[RollbackAuditRecord]:
        """
        Evaluate sliding window health metrics against rollback criteria.
        If any SLA is violated, instantly revert traffic and broadcast alert.

        Returns:
            Optional[RollbackAuditRecord]: Record if rollback was triggered, else None.
        """
        if not self.router.config.is_active or len(self._canary_history) < self.thresholds.min_eval_samples:
            return None

        canary_data = list(self._canary_history)
        latencies = [d[0] for d in canary_data]
        errors = [1 for d in canary_data if d[1]]
        flags = [1 for d in canary_data if d[2]]

        n = len(canary_data)
        p99_latency = float(np.percentile(latencies, 99))
        error_rate = len(errors) / n
        flag_rate = len(flags) / n

        # Champion baseline flag rate
        champ_data = list(self._champion_history)
        champ_flag_rate = (sum(1 for d in champ_data if d[2]) / len(champ_data)) if champ_data else 0.025

        trigger_reasons = []

        if p99_latency > self.thresholds.max_latency_p99_ms:
            trigger_reasons.append(
                f"Canary p99 latency ({p99_latency:.2f}ms) breached {self.thresholds.max_latency_p99_ms}ms SLA"
            )

        if error_rate > self.thresholds.max_error_rate:
            trigger_reasons.append(
                f"Canary error rate ({error_rate*100:.2f}%) exceeded {self.thresholds.max_error_rate*100:.2f}% threshold"
            )

        if champ_flag_rate > 0 and flag_rate > (champ_flag_rate * self.thresholds.max_flag_rate_multiplier):
            trigger_reasons.append(
                f"Canary false-positive rate ({flag_rate*100:.2f}%) spiked >{self.thresholds.max_flag_rate_multiplier}x champion baseline ({champ_flag_rate*100:.2f}%)"
            )

        if trigger_reasons:
            canary_v = self.router.config.canary_version or "unknown"
            champ_v = self.router.config.champion_version
            reason_str = "; ".join(trigger_reasons)

            # 1. Execute atomic rollback
            self.router.reset_to_champion()

            record = RollbackAuditRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                canary_version=canary_v,
                champion_version=champ_v,
                reason=reason_str,
                metrics_at_trigger={
                    "p99_latency_ms": round(p99_latency, 2),
                    "error_rate": round(error_rate, 4),
                    "flag_rate": round(flag_rate, 4),
                    "champ_flag_rate": round(champ_flag_rate, 4),
                },
            )
            self.rollback_history.append(record)

            logger.error(
                "CIRCUIT BREAKER TRIPPED: Automated rollback executed",
                canary=canary_v,
                champion=champ_v,
                reason=reason_str,
            )

            # 2. Broadcast live alert via SSE to all analyst terminals
            await self.streamer.broadcast_event(
                event_type="canary_rollback",
                payload={
                    "canary_version": canary_v,
                    "champion_version": champ_v,
                    "reason": reason_str,
                    "status": "rolled_back",
                },
            )
            return record

        return None

    def get_status_report(self) -> Dict[str, Any]:
        """Generate current canary health and latency report."""
        canary_data = list(self._canary_history)
        if canary_data:
            latencies = [d[0] for d in canary_data]
            p99 = float(np.percentile(latencies, 99))
            p50 = float(np.percentile(latencies, 50))
            err_rate = sum(1 for d in canary_data if d[1]) / len(canary_data)
            flag_rate = sum(1 for d in canary_data if d[2]) / len(canary_data)
        else:
            p99, p50, err_rate, flag_rate = 0.0, 0.0, 0.0, 0.0

        return {
            "canary_active": self.router.config.is_active,
            "canary_version": self.router.config.canary_version,
            "champion_version": self.router.config.champion_version,
            "canary_percentage": self.router.config.canary_percentage,
            "shadow_mode": self.router.config.is_shadow_mode,
            "samples_in_window": len(canary_data),
            "canary_p50_latency_ms": round(p50, 2),
            "canary_p99_latency_ms": round(p99, 2),
            "canary_error_rate": round(err_rate, 4),
            "canary_flag_rate": round(flag_rate, 4),
            "total_rollbacks_recorded": len(self.rollback_history),
        }


_global_rollback_manager: Optional[AutoRollbackManager] = None


def get_rollback_manager() -> AutoRollbackManager:
    """Retrieve shared AutoRollbackManager singleton instance."""
    global _global_rollback_manager
    if _global_rollback_manager is None:
        _global_rollback_manager = AutoRollbackManager()
    return _global_rollback_manager
