"""
SentinelForge — ml/evaluation/regression_eval.py

PURPOSE:
    Gate 2 Evaluation: Head-to-Head Regression Testing Suite.
    Directly compares Candidate Model (M_cand) against active Production Champion (M_prod)
    on an identical test distribution. Candidate MUST meet or exceed production on PR-AUC
    and Recall@1% FPR without violating the sub-10ms latency budget.

ARCHITECTURE POSITION:
    Layer 9 (Evaluation Harness) → Gate 2 Regression Gatekeeper.

INTERVIEW TALKING POINT:
    "Gate 2 is the critical differentiator of SentinelForge. Many ML platforms promote
    models simply because they beat an arbitrary static benchmark. SentinelForge requires
    an automated head-to-head showdown on an identical held-out production slice. If the
    candidate regresses on PR-AUC or Recall@1% FPR—even if overall accuracy looks higher—the
    promotion is automatically blocked to safeguard operational risk operations."
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ml.evaluation.metrics import FraudMetricsCalculator
from shared.logging.logger import get_logger

logger = get_logger("ml.evaluation.regression")


@dataclass
class RegressionTolerances:
    """Allowed tolerances for candidate vs production comparisons."""
    min_delta_pr_auc: float = 0.0000          # Candidate must not regress on PR-AUC
    min_delta_auc_roc: float = -0.0050        # Max allowable drop on ROC-AUC
    min_delta_recall_at_1_fpr: float = 0.0000 # Candidate must not lose recall at 1% FPR
    max_latency_regression_ms: float = 1.0    # Candidate p99 latency cannot exceed prod by >1ms
    hard_latency_cap_ms: float = 10.0         # Hard SLA ceiling


@dataclass
class RegressionResult:
    """Outcome of Head-to-Head Regression Evaluation."""
    is_promotable: bool
    candidate_metrics: Dict[str, Any]
    production_metrics: Dict[str, Any]
    deltas: Dict[str, float]
    regression_reasons: List[str] = field(default_factory=list)


class RegressionEvaluator:
    """
    Executes Head-to-Head showdown between candidate and production champion models.
    """

    def __init__(self, tolerances: Optional[RegressionTolerances] = None) -> None:
        """
        Initialize RegressionEvaluator.

        Args:
            tolerances: Regression thresholds and limits.
        """
        self.tolerances = tolerances or RegressionTolerances()

    def compare_models(
        self,
        y_true: np.ndarray,
        candidate_probs: np.ndarray,
        production_probs: np.ndarray,
        candidate_latency_p99_ms: float = 4.5,
        production_latency_p99_ms: float = 4.3,
        threshold: float = 0.50,
    ) -> RegressionResult:
        """
        Perform head-to-head comparison on identical test traffic slice.

        Args:
            y_true: Ground truth binary labels [N].
            candidate_probs: Candidate model predictions [N].
            production_probs: Production model predictions [N].
            candidate_latency_p99_ms: Measured candidate p99 latency.
            production_latency_p99_ms: Measured production p99 latency.
            threshold: Decision cutoff.

        Returns:
            RegressionResult: Promotion decision, deltas, and diagnostic breakdown.
        """
        cand_metrics = FraudMetricsCalculator.compute_all_metrics(
            y_true=y_true, y_pred_probs=candidate_probs, threshold=threshold
        )
        cand_metrics["latency_p99_ms"] = candidate_latency_p99_ms

        prod_metrics = FraudMetricsCalculator.compute_all_metrics(
            y_true=y_true, y_pred_probs=production_probs, threshold=threshold
        )
        prod_metrics["latency_p99_ms"] = production_latency_p99_ms

        # Compute Deltas (Candidate - Production)
        delta_auc_roc = round(cand_metrics["auc_roc"] - prod_metrics["auc_roc"], 4)
        delta_pr_auc = round(cand_metrics["pr_auc"] - prod_metrics["pr_auc"], 4)
        delta_recall_1_fpr = round(
            cand_metrics["recall_at_1_fpr"] - prod_metrics["recall_at_1_fpr"], 4
        )
        delta_f1 = round(cand_metrics["f1_score"] - prod_metrics["f1_score"], 4)
        delta_latency = round(candidate_latency_p99_ms - production_latency_p99_ms, 3)

        deltas = {
            "delta_auc_roc": delta_auc_roc,
            "delta_pr_auc": delta_pr_auc,
            "delta_recall_at_1_fpr": delta_recall_1_fpr,
            "delta_f1_score": delta_f1,
            "delta_latency_p99_ms": delta_latency,
        }

        # Check for regressions
        regression_reasons: List[str] = []

        if delta_pr_auc < self.tolerances.min_delta_pr_auc:
            regression_reasons.append(
                f"PR-AUC regressed by {delta_pr_auc} (Cand: {cand_metrics['pr_auc']} vs Prod: {prod_metrics['pr_auc']})"
            )

        if delta_auc_roc < self.tolerances.min_delta_auc_roc:
            regression_reasons.append(
                f"AUC-ROC dropped by {delta_auc_roc} (Cand: {cand_metrics['auc_roc']} vs Prod: {prod_metrics['auc_roc']})"
            )

        if delta_recall_1_fpr < self.tolerances.min_delta_recall_at_1_fpr:
            regression_reasons.append(
                f"Recall@1% FPR dropped by {delta_recall_1_fpr} (Cand: {cand_metrics['recall_at_1_fpr']} vs Prod: {prod_metrics['recall_at_1_fpr']})"
            )

        if candidate_latency_p99_ms > self.tolerances.hard_latency_cap_ms:
            regression_reasons.append(
                f"Candidate p99 latency {candidate_latency_p99_ms}ms breaches hard 10ms SLA"
            )

        if delta_latency > self.tolerances.max_latency_regression_ms:
            regression_reasons.append(
                f"Candidate p99 latency degraded by {delta_latency}ms compared to production"
            )

        is_promotable = len(regression_reasons) == 0

        if is_promotable:
            logger.info(
                "Gate 2 Regression Showdown: APPROVED FOR PROMOTION",
                delta_pr_auc=delta_pr_auc,
                delta_recall_1_fpr=delta_recall_1_fpr,
            )
        else:
            logger.warning(
                "Gate 2 Regression Showdown: REJECTED",
                reasons=regression_reasons,
            )

        return RegressionResult(
            is_promotable=is_promotable,
            candidate_metrics=cand_metrics,
            production_metrics=prod_metrics,
            deltas=deltas,
            regression_reasons=regression_reasons,
        )
