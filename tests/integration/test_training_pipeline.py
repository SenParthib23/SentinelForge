"""
SentinelForge — tests/integration/test_training_pipeline.py

PURPOSE:
    Integration test verifying end-to-end LightGBM model training, threshold discovery,
    and artifact export.
"""

import json
from pathlib import Path

import lightgbm as lgb
import pytest

from ml.training.train_lightgbm import train_model


def test_end_to_end_training_pipeline(tmp_path: Path) -> None:
    """Verify complete training run produces valid booster and metadata artifacts."""
    out_dir = tmp_path / "model_export"

    metadata = train_model(
        model_version="test_cand_v1",
        data_path="data/historical_transactions_100k.parquet",
        output_dir=str(out_dir),
        tune_hyperparameters=False,
        num_boost_round=15,  # Fast run for integration test
    )

    # 1. Verify returned metadata
    assert metadata["model_version"] == "test_cand_v1"
    assert "test_metrics" in metadata
    assert metadata["test_metrics"]["auc_roc"] > 0.70
    assert metadata["optimal_threshold"] > 0.0

    # 2. Verify exported artifacts exist
    model_file = out_dir / "test_cand_v1.txt"
    meta_file = out_dir / "test_cand_v1_metadata.json"

    assert model_file.exists()
    assert meta_file.exists()

    # 3. Verify booster can be loaded and predict
    booster = lgb.Booster(model_file=str(model_file))
    assert booster.num_feature() == 17

    # 4. Verify metadata JSON can be parsed
    with open(meta_file, "r") as f:
        loaded_meta = json.load(f)
    assert loaded_meta["model_version"] == "test_cand_v1"
