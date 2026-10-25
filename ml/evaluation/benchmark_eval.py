"""
SentinelForge — ml/evaluation/benchmark_eval.py

PURPOSE:
    Gate 1 Evaluation: Validates candidate models against absolute operational thresholds
    (AUC-ROC >= 0.85, PR-AUC >= 0.70, Recall@1% FPR >= 0.60, Latency p99 < 10ms).

ARCHITECTURE POSITION:
    Layer 9 (Evaluation Harness) → Gate 1 Benchmark Gatekeeper.

INTERVIEW TALKING POINT:
    "Gate 1 prevents broken or severely undertrained models from ever reaching stage 2.
    It checks absolute baseline viability: if a model cannot hit 0.85 AUC-ROC or breaches
    the 10ms p99 latency SLA on standalone benchmarks, it is instantly rejected before any
    regression tests against production traffic are scheduled."
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ml.evaluation.metrics import FraudMetricsCalculator
from shared.logging.logger import get_logger

logger = get_logger("ml.evaluation.benchmark")


@dataclass
class BenchmarkThresholds:
    """Operational criteria for absolute benchmark pass/fail."""
    min_auc_roc: float = 0.85
    min_pr_auc: float = 0.70
    min_recall_at_1_fpr: float = 0.60
    max_latency_p99_ms: float = 10.0


@dataclass
class BenchmarkResult:
    """Outcome of Gate 1 benchmark evaluation."""
    passed: bool
    metrics: Dict[str, Any]
    violations: List[str] = field(default_factory=list)


class BenchmarkEvaluator:
    """
    Evaluates candidate models against hard operational minimum performance criteria.
    """

    def __init__(self, thresholds: Optional[BenchmarkThresholds] = None) -> None:
        """
        Initialize BenchmarkEvaluator.

        Args:
            thresholds: Benchmark criteria thresholds (uses defaults if None).
        """
        self.thresholds = thresholds or BenchmarkThresholds()

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        latency_p99_ms: float = 4.5,
        threshold: float = 0.50,
    ) -> BenchmarkResult:
        """
        Assess candidate predictions against Gate 1 criteria.

        Args:
            y_true: Ground truth binary labels.
            y_pred_probs: Predicted probabilities.
            latency_p99_ms: Measured 99th percentile inference latency.
            threshold: Decision cutoff.

        Returns:
            BenchmarkResult: Pass/fail status, metrics dictionary, and violation list.
        """
        metrics = FraudMetricsCalculator.compute_all_metrics(
            y_true=y_true, y_pred_probs=y_pred_probs, threshold=threshold
        )
        metrics["latency_p99_ms"] = latency_p99_ms

        violations: List[str] = []

        if metrics["auc_roc"] < self.thresholds.min_auc_roc:
            violations.append(
                f"AUC-ROC {metrics['auc_roc']} < required {self.thresholds.min_auc_roc}"
            )

        if metrics["pr_auc"] < self.thresholds.min_pr_auc:
            violations.append(
                f"PR-AUC {metrics['pr_auc']} < required {self.thresholds.min_pr_auc}"
            )

        if metrics["recall_at_1_fpr"] < self.thresholds.min_recall_at_1_fpr:
            violations.append(
                f"Recall@1% FPR {metrics['recall_at_1_fpr']} < required {self.thresholds.min_recall_at_1_fpr}"
            )

        if latency_p99_ms > self.thresholds.max_latency_p99_ms:
            violations.append(
                f"Latency p99 {latency_p99_ms}ms > required {self.thresholds.max_latency_p99_ms}ms"
            )

        passed = len(violations) == 0
        if passed:
            logger.info("Gate 1 Benchmark PASSED", auc=metrics["auc_roc"], pr_auc=metrics["pr_auc"])
        else:
            logger.warning("Gate 1 Benchmark FAILED", violations=violations)

        return BenchmarkResult(passed=passed, metrics=metrics, violations=violations)
