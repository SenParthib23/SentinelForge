"""
SentinelForge — ml/training/feature_builder.py

PURPOSE:
    Transforms raw transaction DataFrames into aligned numerical feature matrices
    using the Single Source of Truth feature definitions. Prevents training-serving skew.

ARCHITECTURE POSITION:
    Layer 7 (Model Training Pipeline) → Feature Matrix Construction.

INTERVIEW TALKING POINT:
    "FeatureBuilder delegates batch feature generation to OfflineFeatureStore, which
    shares the exact same mathematical formulas (Haversine distance, sliding window counts,
    spend z-scores) as the streaming Redis feature engine. Columns are strictly pinned
    to FEATURE_COLUMN_ORDER to ensure complete binary feature alignment during LightGBM inference."
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from services.feature_engine.offline_store import OfflineFeatureStore
from services.scoring.scorer import FEATURE_COLUMN_ORDER
from shared.logging.logger import get_logger

logger = get_logger("ml.training.feature_builder")


class FeatureBuilder:
    """
    Constructs aligned training and evaluation feature matrices from raw transaction tables.
    """

    def __init__(self, offline_store: Optional[OfflineFeatureStore] = None) -> None:
        """
        Initialize FeatureBuilder.

        Args:
            offline_store: OfflineFeatureStore instance for batch feature computation.
        """
        self._offline_store = offline_store or OfflineFeatureStore()

    def build_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Compute batch features across raw transactions.

        Args:
            df_raw: Raw transaction DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing 17 engineered features plus 'is_fraud' label.
        """
        # If df_raw already contains engineered features (e.g., from feedback cache)
        if all(col in df_raw.columns for col in FEATURE_COLUMN_ORDER):
            logger.info("Input already contains engineered features", feature_count=len(FEATURE_COLUMN_ORDER))
            return df_raw

        logger.info("Extracting batch features via OfflineFeatureStore", input_rows=len(df_raw))
        df_features = self._offline_store.extract_batch_features(df_raw)
        logger.info("Batch feature extraction complete", feature_rows=len(df_features))
        return df_features

    def prepare_xy(
        self,
        df_features: pd.DataFrame,
        target_col: str = "is_fraud",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract strictly ordered (X, y) numpy arrays for LightGBM.

        Args:
            df_features: DataFrame containing features and target column.
            target_col: Name of target fraud column.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (X [N x 17], y [N]) float64 arrays.
        """
        # Validate that all required columns are present
        missing = [col for col in FEATURE_COLUMN_ORDER if col not in df_features.columns]
        if missing:
            raise KeyError(f"Missing required feature columns: {missing}")

        X_df = df_features[FEATURE_COLUMN_ORDER].copy()

        # Clean NaN or inf values by replacing with 0.0
        X_df = X_df.fillna(0.0)
        X = np.nan_to_num(X_df.values.astype(np.float64), nan=0.0, posinf=1e6, neginf=-1e6)

        if target_col in df_features.columns:
            y = df_features[target_col].values.astype(np.int32)
        else:
            y = np.zeros(len(df_features), dtype=np.int32)

        return X, y
