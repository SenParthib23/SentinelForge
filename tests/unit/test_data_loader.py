"""
SentinelForge — tests/unit/test_data_loader.py

PURPOSE:
    Unit tests for DataLoader and FeatureBuilder data partitioning and alignment.
"""

import numpy as np
import pandas as pd
import pytest

from ml.training.data_loader import DataLoader
from ml.training.feature_builder import FeatureBuilder
from services.scoring.scorer import FEATURE_COLUMN_ORDER


def test_stratified_split() -> None:
    """Verify stratified split preserves sample proportions and fraud distribution."""
    # Create synthetic dataset with 1000 samples, exactly 30 fraud (3.0%)
    n_samples = 1000
    y = np.zeros(n_samples, dtype=int)
    y[:30] = 1  # 3.0% fraud

    df = pd.DataFrame({
        "txn_id": [f"txn_{i}" for i in range(n_samples)],
        "amount": np.random.uniform(10, 500, n_samples),
        "is_fraud": y,
    })

    loader = DataLoader()
    df_train, df_val, df_test = loader.split_stratified(
        df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, target_col="is_fraud"
    )

    assert len(df_train) == 700
    assert len(df_val) == 150
    assert len(df_test) == 150

    # Verify fraud rate is roughly equal across partitions
    train_rate = df_train["is_fraud"].mean()
    val_rate = df_val["is_fraud"].mean()
    test_rate = df_test["is_fraud"].mean()

    assert np.isclose(train_rate, 0.03, atol=0.01)
    assert np.isclose(val_rate, 0.03, atol=0.01)
    assert np.isclose(test_rate, 0.03, atol=0.01)


def test_feature_builder_prepare_xy() -> None:
    """Verify FeatureBuilder prepares numerical X, y with FEATURE_COLUMN_ORDER."""
    n_samples = 50
    data = {col: np.random.randn(n_samples) for col in FEATURE_COLUMN_ORDER}
    data["is_fraud"] = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    df = pd.DataFrame(data)

    builder = FeatureBuilder()
    X, y = builder.prepare_xy(df)

    assert X.shape == (50, 17)
    assert len(y) == 50
    assert X.dtype == np.float64
    assert y.dtype == np.int32
