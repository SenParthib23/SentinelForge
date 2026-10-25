"""
SentinelForge — ml/training/hyperparameter_search.py

PURPOSE:
    Cross-validated hyperparameter optimization for LightGBM fraud models.
    Tunes tree depth, leaves, sub-sampling, and learning rates to maximize PR-AUC.

ARCHITECTURE POSITION:
    Layer 7 (Model Training Pipeline) → Hyperparameter Tuning.

INTERVIEW TALKING POINT:
    "We optimize hyperparameters against PR-AUC (Area Under Precision-Recall Curve)
    using Stratified K-Fold cross-validation rather than standard ROC-AUC. Under 2.5%
    fraud incidence, ROC-AUC can remain deceptively high (>0.95) even if thousands of
    false positives swamp human risk operations. PR-AUC directly optimizes the trade-off
    that human analysts experience daily."
"""

from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from services.scoring.scorer import FEATURE_COLUMN_ORDER
from shared.logging.logger import get_logger

logger = get_logger("ml.training.hyperparameter_search")


class HyperparameterOptimizer:
    """
    Executes cross-validated hyperparameter grid/random searches for LightGBM.
    """

    def __init__(self, n_splits: int = 3, random_state: int = 42) -> None:
        """
        Initialize HyperparameterOptimizer.

        Args:
            n_splits: Number of stratified cross-validation folds.
            random_state: PRNG seed.
        """
        self.n_splits = n_splits
        self.random_state = random_state

    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scale_pos_weight: float,
        candidate_params: Optional[List[Dict[str, Any]]] = None,
        num_boost_round: int = 100,
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """
        Evaluate candidate parameter dictionaries across stratified folds.

        Args:
            X: Training features array [N x 17].
            y: Training labels array [N].
            scale_pos_weight: Dynamic positive class loss multiplier.
            candidate_params: Optional list of hyperparameter parameter dictionaries.
            num_boost_round: Boosting iterations per fold.

        Returns:
            Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
                (Best params, Best mean PR-AUC, Full history).
        """
        if candidate_params is None:
            candidate_params = [
                {
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_samples": 20,
                },
                {
                    "learning_rate": 0.08,
                    "num_leaves": 63,
                    "max_depth": 7,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "min_child_samples": 15,
                },
                {
                    "learning_rate": 0.03,
                    "num_leaves": 45,
                    "max_depth": 8,
                    "subsample": 0.75,
                    "colsample_bytree": 0.75,
                    "min_child_samples": 30,
                },
            ]

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        history: List[Dict[str, Any]] = []

        best_pr_auc = -1.0
        best_params: Dict[str, Any] = {}

        logger.info(
            "Starting hyperparameter search",
            candidates_count=len(candidate_params),
            cv_folds=self.n_splits,
        )

        for idx, params_cand in enumerate(candidate_params):
            fold_pr_aucs: List[float] = []
            fold_roc_aucs: List[float] = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_COLUMN_ORDER)
                val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLUMN_ORDER, reference=train_data)

                full_params = {
                    "objective": "binary",
                    "metric": ["auc", "binary_logloss"],
                    "boosting_type": "gbdt",
                    "scale_pos_weight": scale_pos_weight,
                    "verbosity": -1,
                    "random_state": self.random_state,
                    **params_cand,
                }

                booster = lgb.train(
                    full_params,
                    train_data,
                    num_boost_round=num_boost_round,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
                )

                preds = booster.predict(X_val)
                fold_pr_aucs.append(float(average_precision_score(y_val, preds)))
                fold_roc_aucs.append(float(roc_auc_score(y_val, preds)))

            mean_pr_auc = float(np.mean(fold_pr_aucs))
            mean_roc_auc = float(np.mean(fold_roc_aucs))

            result = {
                "candidate_idx": idx,
                "params": params_cand,
                "mean_pr_auc": round(mean_pr_auc, 4),
                "mean_roc_auc": round(mean_roc_auc, 4),
            }
            history.append(result)

            logger.info("Candidate evaluated", **result)

            if mean_pr_auc > best_pr_auc:
                best_pr_auc = mean_pr_auc
                best_params = params_cand

        logger.info(
            "Hyperparameter search complete",
            best_pr_auc=round(best_pr_auc, 4),
            best_params=best_params,
        )
        return best_params, best_pr_auc, history
