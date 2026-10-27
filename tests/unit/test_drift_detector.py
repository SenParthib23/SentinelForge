"""
SentinelForge — tests/unit/test_drift_detector.py

PURPOSE:
    Unit tests for DriftDetector PSI (Population Stability Index) and Wasserstein distance.
"""

import numpy as np
import pandas as pd
import pytest

from ml.monitoring.drift_detector import DriftDetector


def test_psi_identical_distributions() -> None:
    """Verify PSI between identical distributions is negligible (< 0.05)."""
    np.random.seed(42)
    ref = np.random.normal(50, 10, 1000)
    cur = np.random.normal(50, 10, 1000)

    detector = DriftDetector()
    psi = detector.compute_feature_psi(ref, cur)
    assert psi < 0.05


def test_psi_severe_drift_detection() -> None:
    """Verify significant distribution shift triggers PSI >= 0.25."""
    np.random.seed(42)
    ref = np.random.normal(50, 5, 1000)
    # Drifted distribution with completely shifted mean and variance
    cur = np.random.normal(120, 25, 1000)

    detector = DriftDetector()
    psi = detector.compute_feature_psi(ref, cur)
    assert psi >= 0.25


def test_compute_drift_report() -> None:
    """Verify drift report correctly categorizes stable vs drifted features."""
    np.random.seed(42)
    n = 500

    ref_df = pd.DataFrame({
        "amount_zscore": np.random.normal(0, 1, n),
        "txn_count_1min": np.random.poisson(2, n),
    })

    # Shifted amount_zscore distribution (severe drift)
    cur_df = pd.DataFrame({
        "amount_zscore": np.random.normal(4.5, 1.5, n),
        "txn_count_1min": np.random.poisson(2, n),  # stable
    })

    detector = DriftDetector(reference_data=ref_df)
    report = detector.compute_drift_report(cur_df)

    assert report.overall_drift_detected is True
    assert "amount_zscore" in report.drifted_features
    assert report.feature_metrics["amount_zscore"].severity == "SEVERE"
    assert report.feature_metrics["txn_count_1min"].severity == "NONE"
