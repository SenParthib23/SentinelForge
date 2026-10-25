"""
SentinelForge — ml/training/train_lightgbm.py

PURPOSE:
    Production LightGBM model retraining pipeline. Ingests historical data, integrates
    fresh human feedback labels from PostgreSQL, handles class imbalance, optimizes
    hyperparameters, trains booster, and exports candidate model artifacts.

ARCHITECTURE POSITION:
    Layer 7 (Model Training Pipeline) → End-to-End Orchestrator.

INTERVIEW TALKING POINT:
    "Our retraining script bridges the human-in-the-loop feedback flywheel: it extracts
    ground-truth labels adjudicated by risk analysts, runs stratified temporal partitioning,
    tunes LightGBM with cost-sensitive weighting, and exports candidate weights ready for
    Layer 9's automated Head-to-Head Regression Suite."
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.training.class_imbalance import ClassImbalanceManager
from ml.training.data_loader import DataLoader
from ml.training.feature_builder import FeatureBuilder
from ml.training.hyperparameter_search import HyperparameterOptimizer
from services.scoring.scorer import FEATURE_COLUMN_ORDER
from shared.logging.logger import configure_logging, get_logger

configure_logging(environment="development", log_level="INFO")
logger = get_logger("ml.training.train_lightgbm")


def train_model(
    model_version: str = "candidate_v2",
    data_path: str = "data/historical_transactions_100k.parquet",
    output_dir: str = "model_cache/lightgbm",
    tune_hyperparameters: bool = False,
    num_boost_round: int = 150,
) -> Dict[str, Any]:
    """
    Train and export a LightGBM fraud model candidate.

    Args:
        model_version: Version identifier (e.g. 'candidate_v2').
        data_path: Path to historical dataset parquet.
        output_dir: Target directory for exported model files.
        tune_hyperparameters: If True, execute cross-validated hyperparameter search.
        num_boost_round: Maximum boosting iterations.

    Returns:
        Dict[str, Any]: Comprehensive training and evaluation metadata.
    """
    logger.info("Initializing LightGBM training run", model_version=model_version)

    # 1. Load Data
    data_loader = DataLoader()
    df_raw = data_loader.load_historical_parquet(data_path)

    # 2. Extract Features via Single Source of Truth
    feature_builder = FeatureBuilder()
    df_features = feature_builder.build_features(df_raw)

    # 3. Stratified Split (70% Train, 15% Val, 15% Test)
    df_train, df_val, df_test = data_loader.split_stratified(
        df_features, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    X_train, y_train = feature_builder.prepare_xy(df_train)
    X_val, y_val = feature_builder.prepare_xy(df_val)
    X_test, y_test = feature_builder.prepare_xy(df_test)

    # 4. Class Imbalance Weighting
    imbalance_mgr = ClassImbalanceManager(cost_false_negative=500.0, cost_false_positive=10.0)
    scale_pos_weight = imbalance_mgr.compute_scale_pos_weight(y_train)

    # 5. Hyperparameter Configuration
    if tune_hyperparameters:
        logger.info("Executing hyperparameter optimization...")
        optimizer = HyperparameterOptimizer(n_splits=3)
        best_params, _, _ = optimizer.optimize(
            X_train, y_train, scale_pos_weight=scale_pos_weight, num_boost_round=80
        )
    else:
        best_params = {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }

    training_params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
        "random_state": 42,
        **best_params,
    }

    # 6. Train Booster with Early Stopping on Validation Set
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLUMN_ORDER)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLUMN_ORDER, reference=train_data)
    test_data = lgb.Dataset(X_test, label=y_test, feature_name=FEATURE_COLUMN_ORDER, reference=train_data)

    logger.info("Fitting LightGBM booster...")
    booster = lgb.train(
        training_params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    # 7. Optimal Threshold Discovery on Validation Set
    val_preds = booster.predict(X_val)
    opt_threshold, cost_breakdown = imbalance_mgr.find_cost_optimal_threshold(y_val, val_preds)

    # 8. Unseen Test Set Evaluation
    test_preds = booster.predict(X_test)
    test_binary_standard = (test_preds >= 0.50).astype(int)
    test_binary_optimal = (test_preds >= opt_threshold).astype(int)

    auc_roc = float(roc_auc_score(y_test, test_preds))
    pr_auc = float(average_precision_score(y_test, test_preds))
    f1_standard = float(f1_score(y_test, test_binary_standard, zero_division=0))
    f1_optimal = float(f1_score(y_test, test_binary_optimal, zero_division=0))
    prec_optimal = float(precision_score(y_test, test_binary_optimal, zero_division=0))
    rec_optimal = float(recall_score(y_test, test_binary_optimal, zero_division=0))

    test_financial_costs = imbalance_mgr.calculate_financial_cost(y_test, test_binary_optimal)

    metadata: Dict[str, Any] = {
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "test_samples": len(y_test),
        "scale_pos_weight": round(scale_pos_weight, 2),
        "optimal_threshold": opt_threshold,
        "parameters": training_params,
        "test_metrics": {
            "auc_roc": round(auc_roc, 4),
            "pr_auc": round(pr_auc, 4),
            "f1_score_optimal": round(f1_optimal, 4),
            "precision_optimal": round(prec_optimal, 4),
            "recall_optimal": round(rec_optimal, 4),
            "f1_score_standard_05": round(f1_standard, 4),
            "financial_loss": test_financial_costs["total_financial_loss"],
            "missed_fraud_count": test_financial_costs["false_negatives"],
            "false_alarms_count": test_financial_costs["false_positives"],
        },
    }

    # 9. Save Artifacts
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{model_version}.txt"
    meta_path = out_dir / f"{model_version}_metadata.json"

    booster.save_model(str(model_path))
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Model training run complete and exported",
        model_version=model_version,
        model_path=str(model_path),
        test_auc=round(auc_roc, 4),
        test_pr_auc=round(pr_auc, 4),
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SentinelForge LightGBM Model")
    parser.add_argument("--version", type=str, default="candidate_v2", help="Model version tag")
    parser.add_argument("--tune", action="store_true", help="Execute hyperparameter tuning")
    args = parser.parse_args()

    train_model(model_version=args.version, tune_hyperparameters=args.tune)
