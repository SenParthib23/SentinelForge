"""
SentinelForge — ml/evaluation/metrics.py

PURPOSE:
    Domain-specific fraud detection metrics calculator. Computes standard classification
    scores (ROC-AUC, PR-AUC, F1) alongside payments industry performance indicators:
    Precision@90% Recall, Recall@1% False Positive Rate, and inference latency percentiles.

ARCHITECTURE POSITION:
    Layer 9 (Evaluation Harness) → Metric Evaluation Engine.

INTERVIEW TALKING POINT:
    "Standard precision and recall do not capture the operational constraints of card rails.
    Payments teams focus on 'Recall@1% FPR' (how much fraud we catch while declining at most
    1% of legitimate customers) and 'Precision@90% Recall' (how noisy the queue is when catching
    90% of chargebacks). We measure both alongside sub-10ms inference latency percentiles."
"""

import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from shared.logging.logger import get_logger

logger = get_logger("ml.evaluation.metrics")


class FraudMetricsCalculator:
    """
    Computes standard, domain-specific, and performance latency metrics for fraud models.
    """

    @staticmethod
    def compute_all_metrics(
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        threshold: float = 0.50,
    ) -> Dict[str, Any]:
        """
        Compute complete suite of fraud classification metrics.

        Args:
            y_true: Binary ground truth labels (0 = legitimate, 1 = fraud).
            y_pred_probs: Predicted fraud probabilities in [0.0, 1.0].
            threshold: Decision cutoff for binary classification.

        Returns:
            Dict[str, Any]: Dictionary containing AUCs, operational points, and confusion values.
        """
        y_pred_binary = (y_pred_probs >= threshold).astype(int)

        auc_roc = float(roc_auc_score(y_true, y_pred_probs))
        pr_auc = float(average_precision_score(y_true, y_pred_probs))
        prec = float(precision_score(y_true, y_pred_binary, zero_division=0))
        rec = float(recall_score(y_true, y_pred_binary, zero_division=0))
        f1 = float(f1_score(y_true, y_pred_binary, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary, labels=[0, 1]).ravel()

        prec_at_90_rec = FraudMetricsCalculator.compute_precision_at_recall(
            y_true, y_pred_probs, target_recall=0.90
        )
        rec_at_1_fpr = FraudMetricsCalculator.compute_recall_at_fpr(
            y_true, y_pred_probs, max_fpr=0.01
        )

        return {
            "auc_roc": round(auc_roc, 4),
            "pr_auc": round(pr_auc, 4),
            "threshold": round(threshold, 3),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "precision_at_90_recall": round(prec_at_90_rec, 4),
            "recall_at_1_fpr": round(rec_at_1_fpr, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "total_samples": len(y_true),
        }

    @staticmethod
    def compute_precision_at_recall(
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        target_recall: float = 0.90,
    ) -> float:
        """
        Compute precision achieved when operating at or above target recall.

        Args:
            y_true: Binary ground truth labels.
            y_pred_probs: Continuous fraud probabilities.
            target_recall: Target recall fraction (default 0.90).

        Returns:
            float: Highest precision achieved while recall >= target_recall.
        """
        precisions, recalls, _ = precision_recall_curve(y_true, y_pred_probs)
        # Find points where recall >= target_recall
        valid_indices = np.where(recalls >= target_recall)[0]
        if len(valid_indices) == 0:
            return 0.0
        return float(np.max(precisions[valid_indices]))

    @staticmethod
    def compute_recall_at_fpr(
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        max_fpr: float = 0.01,
    ) -> float:
        """
        Compute maximum recall achieved while maintaining False Positive Rate <= max_fpr.

        Args:
            y_true: Binary ground truth labels.
            y_pred_probs: Continuous fraud probabilities.
            max_fpr: Maximum acceptable false positive rate (default 0.01 = 1%).

        Returns:
            float: Highest recall achievable within the FPR budget.
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        valid_indices = np.where(fpr <= max_fpr)[0]
        if len(valid_indices) == 0:
            return 0.0
        return float(np.max(tpr[valid_indices]))

    @staticmethod
    def benchmark_inference_latency(
        model_predict_fn,
        sample_feature_vector: np.ndarray,
        iterations: int = 100,
    ) -> Dict[str, float]:
        """
        Benchmark inference execution latency percentiles under repeated execution.

        Args:
            model_predict_fn: Callable taking feature array and returning prediction.
            sample_feature_vector: Feature vector shape [1, 17] or [N, 17].
            iterations: Number of warmup and timed inference runs.

        Returns:
            Dict[str, float]: p50, p95, p99 latency in milliseconds.
        """
        # Warmup runs
        for _ in range(10):
            _ = model_predict_fn(sample_feature_vector)

        latencies_ms = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model_predict_fn(sample_feature_vector)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        latencies_arr = np.array(latencies_ms)
        return {
            "latency_p50_ms": round(float(np.percentile(latencies_arr, 50)), 3),
            "latency_p95_ms": round(float(np.percentile(latencies_arr, 95)), 3),
            "latency_p99_ms": round(float(np.percentile(latencies_arr, 99)), 3),
            "latency_mean_ms": round(float(np.mean(latencies_arr)), 3),
        }
