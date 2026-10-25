"""
SentinelForge — tests/unit/test_class_imbalance.py

PURPOSE:
    Unit tests for ClassImbalanceManager loss weighting and cost-sensitive threshold search.
"""

import numpy as np
import pytest

from ml.training.class_imbalance import ClassImbalanceManager


def test_scale_pos_weight_computation() -> None:
    """Verify scale_pos_weight = N_neg / N_pos."""
    imbalance = ClassImbalanceManager()

    # 95 legitimate, 5 fraud => weight = 95 / 5 = 19.0
    y = np.array([0] * 95 + [1] * 5)
    weight = imbalance.compute_scale_pos_weight(y)
    assert weight == 19.0

    # Fallback when no fraud present
    y_empty = np.zeros(50)
    assert imbalance.compute_scale_pos_weight(y_empty) == 1.0


def test_financial_cost_calculation() -> None:
    """Verify cost formula: FN * 500 + FP * 10."""
    imbalance = ClassImbalanceManager(cost_false_negative=500.0, cost_false_positive=10.0)

    # 2 FN, 3 FP
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_pred = np.array([0, 0, 1, 1, 1, 0])  # Both frauds missed (2 FN), 3 false alarms (3 FP)

    costs = imbalance.calculate_financial_cost(y_true, y_pred)
    assert costs["false_negatives"] == 2
    assert costs["false_positives"] == 3
    # 2 * 500 + 3 * 10 = 1000 + 30 = 1030
    assert costs["total_financial_loss"] == 1030.0


def test_find_cost_optimal_threshold() -> None:
    """Verify search scans probabilities and returns optimal cutoff."""
    imbalance = ClassImbalanceManager()
    np.random.seed(42)

    y_true = np.array([0]*90 + [1]*10)
    # Predicted probabilities where fraud is centered around 0.70 and legit around 0.10
    probs = np.concatenate([
        np.random.beta(1, 9, 90),
        np.random.beta(7, 3, 10),
    ])

    best_thresh, breakdown = imbalance.find_cost_optimal_threshold(y_true, probs)
    assert 0.05 <= best_thresh <= 0.95
    assert "total_financial_loss" in breakdown
    assert "optimal_threshold" in breakdown
