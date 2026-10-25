"""
SentinelForge — tests/unit/test_eval_metrics.py

PURPOSE:
    Unit tests for FraudMetricsCalculator domain metrics and latency benchmarker.
"""

import numpy as np
import pytest

from ml.evaluation.metrics import FraudMetricsCalculator


def test_compute_all_metrics() -> None:
    """Verify AUC-ROC, PR-AUC, F1 and domain metrics calculation."""
    np.random.seed(42)
    y_true = np.array([0]*80 + [1]*20)
    # Good classifier probabilities
    probs = np.concatenate([
        np.random.uniform(0.0, 0.4, 80),
        np.random.uniform(0.6, 1.0, 20),
    ])

    metrics = FraudMetricsCalculator.compute_all_metrics(y_true, probs, threshold=0.50)

    assert metrics["auc_roc"] >= 0.90
    assert metrics["pr_auc"] >= 0.80
    assert metrics["recall"] >= 0.80
    assert metrics["precision"] >= 0.80
    assert "precision_at_90_recall" in metrics
    assert "recall_at_1_fpr" in metrics
    assert metrics["total_samples"] == 100


def test_recall_at_fpr_and_precision_at_recall() -> None:
    """Verify bounds of domain metrics."""
    y_true = np.array([0]*90 + [1]*10)
    probs = np.linspace(0.01, 0.99, 100)

    prec_90 = FraudMetricsCalculator.compute_precision_at_recall(y_true, probs, target_recall=0.90)
    assert 0.0 <= prec_90 <= 1.0

    rec_1 = FraudMetricsCalculator.compute_recall_at_fpr(y_true, probs, max_fpr=0.01)
    assert 0.0 <= rec_1 <= 1.0


def test_benchmark_inference_latency() -> None:
    """Verify latency benchmark returns p50, p95, p99."""
    def dummy_predict(X):
        return np.array([0.5])

    sample = np.zeros((1, 17))
    latencies = FraudMetricsCalculator.benchmark_inference_latency(dummy_predict, sample, iterations=20)

    assert "latency_p50_ms" in latencies
    assert "latency_p95_ms" in latencies
    assert "latency_p99_ms" in latencies
    assert latencies["latency_p99_ms"] >= latencies["latency_p50_ms"]
