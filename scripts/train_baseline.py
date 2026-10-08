"""
SentinelForge — scripts/train_baseline.py

PURPOSE:
    Trains the initial baseline LightGBM fraud detection model on historical
    transactions, computes benchmark evaluation metrics (AUC, Precision, Recall, F1),
    and exports model weights to `model_cache/lightgbm/baseline_v1.txt`.

ARCHITECTURE POSITION:
    Phase 4 Baseline Model & Phase 7 ML Training Pipeline Foundation.

CLASS IMBALANCE HANDLING:
    Because payment fraud accounts for only ~2.5% of total volume, standard binary
    cross-entropy loss yields high accuracy by simply predicting 0 (legitimate) on all inputs.
    We compute dynamic `scale_pos_weight = N_legitimate / N_fraud` (typically ~38.0)
    to penalize false negatives proportionally, optimizing the precision-recall frontier.

INTERVIEW TALKING POINT:
    "We trained a LightGBM GBDT with `scale_pos_weight` tuned to the empirical
    class imbalance ratio (~38.0). This gives the gradient boosting trees sufficient
    loss sensitivity on sparse positive fraud events without requiring synthetic SMOTE
    oversampling, which can introduce artifacts in high-dimensional sliding window features."
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from services.feature_engine.offline_store import OfflineFeatureStore
from services.scoring.scorer import FEATURE_COLUMN_ORDER
from shared.config.settings import get_settings
from shared.logging.logger import configure_logging, get_logger

configure_logging(environment="development", log_level="INFO")
logger = get_logger("train_baseline")


def train_baseline_model(
    data_path: str = "data/historical_transactions_100k.parquet",
    output_dir: str = "model_cache/lightgbm",
) -> None:
    """
    Train and export baseline LightGBM fraud classifier.

    Args:
        data_path: Path to historical transactions Parquet file.
        output_dir: Model export destination directory.
    """
    settings = get_settings()
    data_file = Path(data_path)

    if not data_file.exists():
        alt_file = Path("data/historical_transactions_10k.parquet")
        if alt_file.exists():
            data_file = alt_file
        else:
            raise FileNotFoundError(f"Historical dataset not found at {data_path}")

    logger.info("Loading transaction dataset", path=str(data_file))
    df_raw = pd.read_parquet(data_file)
    logger.info("Loaded raw transactions", count=len(df_raw))

    # 1. Batch Feature Extraction using Single Source of Truth
    logger.info("Extracting batch features via OfflineFeatureStore...")
    offline_store = OfflineFeatureStore()
    df_features = offline_store.extract_batch_features(df_raw)
    logger.info("Batch feature extraction complete", feature_rows=len(df_features))

    X = df_features[FEATURE_COLUMN_ORDER].values
    y = df_features["is_fraud"].values

    # 2. Stratified Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = float(n_neg / max(1, n_pos))

    logger.info(
        "Class distribution",
        train_legitimate=int(n_neg),
        train_fraud=int(n_pos),
        scale_pos_weight=round(scale_pos_weight, 2),
    )

    # 3. LightGBM Dataset & Training Configuration
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLUMN_ORDER)
    test_data = lgb.Dataset(
        X_test, label=y_test, feature_name=FEATURE_COLUMN_ORDER, reference=train_data
    )

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "scale_pos_weight": scale_pos_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": -1,
        "random_state": 42,
    }

    logger.info("Training LightGBM model...")
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=150,
        valid_sets=[train_data, test_data],
        valid_names=["train", "test"],
    )

    # 4. Evaluation on Test Set
    y_pred_probs = booster.predict(X_test)
    y_pred_binary = (y_pred_probs >= 0.50).astype(int)

    auc_roc = float(roc_auc_score(y_test, y_pred_probs))
    pr_auc = float(average_precision_score(y_test, y_pred_probs))
    prec = float(precision_score(y_test, y_pred_binary, zero_division=0))
    rec = float(recall_score(y_test, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_test, y_pred_binary, zero_division=0))

    metrics = {
        "model_version": "baseline_v1",
        "auc_roc": round(auc_roc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "scale_pos_weight": round(scale_pos_weight, 2),
        "test_samples": len(y_test),
        "test_fraud_count": int(np.sum(y_test == 1)),
    }

    logger.info("Baseline Evaluation Metrics", **metrics)

    # 5. Export Model Artifacts
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    baseline_file = out_path / "baseline_v1.txt"
    v1_file = out_path / "v1.txt"
    eval_file = out_path / "baseline_v1_eval.json"

    booster.save_model(str(baseline_file))
    booster.save_model(str(v1_file))

    with open(eval_file, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "Successfully saved baseline model artifacts",
        model_path=str(baseline_file),
        eval_path=str(eval_file),
    )


if __name__ == "__main__":
    train_baseline_model()
