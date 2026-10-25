"""
SentinelForge — tests/unit/test_regression_eval.py

PURPOSE:
    Unit tests for Gate 1 BenchmarkEvaluator and Gate 2 RegressionEvaluator.
"""

import numpy as np
import pytest

from ml.evaluation.benchmark_eval import BenchmarkEvaluator, BenchmarkThresholds
from ml.evaluation.regression_eval import RegressionEvaluator, RegressionTolerances


def test_gate1_benchmark_pass_and_fail() -> None:
    """Verify Gate 1 passes strong models and rejects models failing operational thresholds."""
    y_true = np.array([0]*90 + [1]*10)

    # 1. Strong model predictions
    good_probs = np.concatenate([np.random.uniform(0.0, 0.2, 90), np.random.uniform(0.8, 1.0, 10)])
    evaluator = BenchmarkEvaluator()
    res_pass = evaluator.evaluate(y_true, good_probs, latency_p99_ms=4.2)
    assert res_pass.passed is True
    assert len(res_pass.violations) == 0

    # 2. Failing model (poor AUC and high latency)
    bad_probs = np.random.uniform(0.4, 0.6, 100)
    res_fail = evaluator.evaluate(y_true, bad_probs, latency_p99_ms=15.0)
    assert res_fail.passed is False
    assert any("AUC-ROC" in v for v in res_fail.violations)
    assert any("Latency" in v for v in res_fail.violations)


def test_gate2_regression_showdown() -> None:
    """Verify Gate 2 promotes superior candidate and blocks regressing candidate."""
    y_true = np.array([0]*90 + [1]*10)

    # Base production model
    prod_probs = np.concatenate([np.random.uniform(0.0, 0.3, 90), np.random.uniform(0.7, 0.9, 10)])

    # Scenario A: Superior Candidate (higher PR-AUC and Recall)
    cand_good = np.concatenate([np.random.uniform(0.0, 0.15, 90), np.random.uniform(0.85, 1.0, 10)])
    evaluator = RegressionEvaluator()

    res_good = evaluator.compare_models(
        y_true=y_true,
        candidate_probs=cand_good,
        production_probs=prod_probs,
        candidate_latency_p99_ms=4.4,
        production_latency_p99_ms=4.5,
    )
    assert res_good.is_promotable is True
    assert res_good.deltas["delta_pr_auc"] >= 0.0

    # Scenario B: Regressed Candidate (lower PR-AUC and breaches 10ms latency)
    cand_bad = np.random.uniform(0.3, 0.6, 100)
    res_bad = evaluator.compare_models(
        y_true=y_true,
        candidate_probs=cand_bad,
        production_probs=prod_probs,
        candidate_latency_p99_ms=12.5,
        production_latency_p99_ms=4.5,
    )
    assert res_bad.is_promotable is False
    assert len(res_bad.regression_reasons) >= 1
