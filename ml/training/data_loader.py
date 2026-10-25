"""
SentinelForge — ml/training/data_loader.py

PURPOSE:
    Loads and partitions financial transaction datasets for offline ML training.
    Supports ingesting historical Parquet datasets and merging fresh ground-truth
    labels from the PostgreSQL `labeled_data` feedback store.

ARCHITECTURE POSITION:
    Layer 7 (Model Training Pipeline) → Ingestion & Data Preparation.

INTERVIEW TALKING POINT:
    "Our DataLoader supports continuous learning by querying newly labeled human feedback
    from the PostgreSQL `labeled_data` table and merging it into the historical Parquet
    training corpus. Stratified splitting preserves our ~2.5% fraud distribution identically
    across train, validation, and held-out test partitions."
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from shared.db.postgres import PostgresClient, get_postgres_client
from shared.logging.logger import get_logger

logger = get_logger("ml.training.data_loader")


class DataLoader:
    """
    Ingests and partitions transaction data for classical ML model training.
    """

    def __init__(self, postgres_client: Optional[PostgresClient] = None) -> None:
        """
        Initialize DataLoader.

        Args:
            postgres_client: Optional PostgresClient for querying analyst feedback.
        """
        self._postgres = postgres_client or get_postgres_client()

    async def load_analyst_feedback(self) -> pd.DataFrame:
        """
        Query verified analyst adjudications from the PostgreSQL `labeled_data` table.

        Returns:
            pd.DataFrame: DataFrame containing verified transaction IDs, feature snapshots,
                          and labels, or an empty DataFrame if no feedback is available.
        """
        try:
            if self._postgres.pool is not None or await self._postgres.is_healthy():
                query = """
                SELECT transaction_id, features, risk_score, analyst_label, created_at
                FROM labeled_data
                ORDER BY created_at ASC;
                """
                records = await self._postgres.fetch(query)
                if records:
                    rows = []
                    for r in records:
                        features = r["features"]
                        if isinstance(features, str):
                            features = json.loads(features)
                        row = dict(features)
                        row["transaction_id"] = r["transaction_id"]
                        row["is_fraud"] = 1 if r["analyst_label"] == "fraud" else 0
                        rows.append(row)
                    df_feedback = pd.DataFrame(rows)
                    logger.info("Loaded analyst feedback records from PostgreSQL", count=len(df_feedback))
                    return df_feedback
        except Exception as exc:
            logger.debug("Could not load analyst feedback from PostgreSQL", error=str(exc))

        return pd.DataFrame()

    def load_historical_parquet(self, parquet_path: str = "data/historical_transactions_100k.parquet") -> pd.DataFrame:
        """
        Load historical transactions from Parquet file.

        Args:
            parquet_path: Primary path to historical Parquet file.

        Returns:
            pd.DataFrame: Raw historical transactions.
        """
        data_file = Path(parquet_path)
        if not data_file.exists():
            alt_file = Path("data/historical_transactions_10k.parquet")
            if alt_file.exists():
                data_file = alt_file
            else:
                raise FileNotFoundError(f"Historical dataset not found at {parquet_path}")

        logger.info("Loading historical transaction dataset", path=str(data_file))
        df = pd.read_parquet(data_file)
        logger.info("Loaded historical transactions", rows=len(df))
        return df

    def split_stratified(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        target_col: str = "is_fraud",
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute stratified 3-way split (Train / Validation / Test).

        Args:
            df: DataFrame to split.
            train_ratio: Proportion of training data (e.g. 0.70).
            val_ratio: Proportion of validation data (e.g. 0.15).
            test_ratio: Proportion of test data (e.g. 0.15).
            target_col: Label column name for stratification.
            random_state: PRNG seed for reproducibility.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (Train, Val, Test) DataFrames.
        """
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame columns")

        # First split: Train vs (Val + Test)
        val_test_ratio = val_ratio + test_ratio
        df_train, df_val_test = train_test_split(
            df,
            test_size=val_test_ratio,
            random_state=random_state,
            stratify=df[target_col],
        )

        # Second split: Val vs Test
        test_relative_ratio = test_ratio / val_test_ratio
        df_val, df_test = train_test_split(
            df_val_test,
            test_size=test_relative_ratio,
            random_state=random_state,
            stratify=df_val_test[target_col],
        )

        logger.info(
            "Stratified split complete",
            train_count=len(df_train),
            val_count=len(df_val),
            test_count=len(df_test),
            train_fraud_rate=float(df_train[target_col].mean()),
            val_fraud_rate=float(df_val[target_col].mean()),
            test_fraud_rate=float(df_test[target_col].mean()),
        )

        return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)
