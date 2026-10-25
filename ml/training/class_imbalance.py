"""
SentinelForge — ml/training/class_imbalance.py

PURPOSE:
    Provides mathematical class imbalance weighting and cost-sensitive threshold optimization.
    Addresses the extreme ~2.5% fraud class imbalance without resorting to synthetic SMOTE artifacts.

ARCHITECTURE POSITION:
    Layer 7 (Model Training Pipeline) → Loss Weighting & Cost-Sensitive Optimization.

INTERVIEW TALKING POINT:
    "In credit card fraud detection, accuracy is a vanity metric because predicting all zeros
    yields 97.5% accuracy while losing millions in fraudulent transactions. A False Negative
    (missed fraud) costs ~$500 in chargebacks and fines, whereas a False Positive (alerting a
    legitimate cardholder) costs ~$10 in customer friction. We dynamically compute `scale_pos_weight`
    for gradient boosting loss and optimize the operational decision threshold to minimize
    financial cost on validation data."
"""

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix

from shared.logging.logger import get_logger

logger = get_logger("ml.training.class_imbalance")


class ClassImbalanceManager:
    """
    Manages loss weighting and cost-sensitive threshold selection for fraud models.
    """

    def __init__(
        self,
        cost_false_negative: float = 500.0,
        cost_false_positive: float = 10.0,
    ) -> None:
        """
        Initialize ClassImbalanceManager.

        Args:
            cost_false_negative: Financial loss per missed fraud event ($500 typical).
            cost_false_positive: Operational cost per false alarm ($10 typical).
        """
        self.cost_fn = cost_false_negative
        self.cost_fp = cost_false_positive

    def compute_scale_pos_weight(self, y: np.ndarray) -> float:
        """
        Compute dynamic LightGBM scale_pos_weight = N_neg / max(1, N_pos).

        Args:
            y: Array of binary fraud labels (0 = legitimate, 1 = fraud).

        Returns:
            float: Weight factor for positive class in gradient computation.
        """
        n_pos = int(np.sum(y == 1))
        n_neg = int(np.sum(y == 0))

        if n_pos == 0:
            logger.warning("No positive fraud events detected in training batch")
            return 1.0

        weight = float(n_neg / n_pos)
        logger.info(
            "Computed dynamic scale_pos_weight",
            legitimate_count=n_neg,
            fraud_count=n_pos,
            scale_pos_weight=round(weight, 2),
        )
        return weight

    def calculate_financial_cost(
        self,
        y_true: np.ndarray,
        y_pred_binary: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute financial loss from confusion matrix.

        Args:
            y_true: Ground truth binary labels.
            y_pred_binary: Binary predictions (0 or 1).

        Returns:
            Dict[str, Any]: Financial loss and confusion matrix breakdown.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()
        total_loss = (fn * self.cost_fn) + (fp * self.cost_fp)

        return {
            "total_financial_loss": float(total_loss),
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "cost_per_transaction": float(total_loss / max(1, len(y_true))),
        }

    def find_cost_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        threshold_steps: int = 100,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Scan probability thresholds to identify the cutoff that minimizes financial loss.

        Args:
            y_true: Ground truth binary labels.
            y_pred_probs: Continuous predicted fraud probabilities [0, 1].
            threshold_steps: Number of search steps between 0.05 and 0.95.

        Returns:
            Tuple[float, Dict[str, Any]]: (Optimal threshold, Detailed cost breakdown).
        """
        candidate_thresholds = np.linspace(0.05, 0.95, threshold_steps)
        best_threshold = 0.50
        min_cost = float("inf")
        best_breakdown: Dict[str, Any] = {}

        for t in candidate_thresholds:
            y_binary = (y_pred_probs >= t).astype(int)
            res = self.calculate_financial_cost(y_true, y_binary)
            if res["total_financial_loss"] < min_cost:
                min_cost = res["total_financial_loss"]
                best_threshold = float(t)
                best_breakdown = res

        best_breakdown["optimal_threshold"] = round(best_threshold, 3)
        logger.info(
            "Identified cost-optimal threshold",
            optimal_threshold=best_breakdown["optimal_threshold"],
            total_loss=best_breakdown["total_financial_loss"],
            fn=best_breakdown["false_negatives"],
            fp=best_breakdown["false_positives"],
        )
        return best_threshold, best_breakdown
