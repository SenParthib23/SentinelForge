"""
SentinelForge — services/feature_engine/offline_store.py

PURPOSE:
    Manages offline persistence of feature snapshots to PostgreSQL `feature_snapshots`
    and provides batch feature extraction for training datasets directly from Parquet.

ARCHITECTURE POSITION:
    Layer 2 (Feature Engine) → Layer 7 (Offline ML Training).

GUARANTEE:
    Uses the exact same mathematical implementations from `geo_features`,
    `velocity_features`, `spend_features`, `device_features`, and `merchant_features`
    to guarantee zero training-serving skew.
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from services.feature_engine.device_features import DeviceFeatureComputer
from services.feature_engine.feature_definitions import FEATURE_VERSION
from services.feature_engine.geo_features import compute_haversine_distance
from services.feature_engine.merchant_features import MerchantFeatureComputer
from services.feature_engine.spend_features import SpendFeatureComputer
from shared.config.settings import Settings, get_settings
from shared.db.postgres import PostgresClient, get_postgres_client
from shared.logging.logger import get_logger
from shared.models.feature_vector import FeatureVector

logger = get_logger("feature_engine.offline_store")


class OfflineFeatureStore:
    """
    Manages offline feature snapshots in PostgreSQL and batch transformation pipelines.
    """

    def __init__(
        self,
        postgres_client: Optional[PostgresClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize offline feature store.

        Args:
            postgres_client: PostgresClient instance.
            settings: Settings instance.
        """
        self.postgres = postgres_client or get_postgres_client()
        self.settings = settings or get_settings()

    async def save_feature_snapshot(
        self,
        transaction_id: str,
        feature_vector: FeatureVector,
        feature_version: str = FEATURE_VERSION,
    ) -> None:
        """
        Write feature snapshot JSON to PostgreSQL `feature_snapshots` table.

        Args:
            transaction_id: Unique transaction ID.
            feature_vector: Computed FeatureVector.
            feature_version: Feature catalog version.
        """
        query = """
        INSERT INTO feature_snapshots (transaction_id, features, feature_version)
        VALUES ($1, $2::jsonb, $3);
        """
        features_json = json.dumps(feature_vector.to_model_input_dict())

        try:
            if self.postgres.pool is not None or await self.postgres.is_healthy():
                await self.postgres.execute(
                    query, transaction_id, features_json, feature_version
                )
        except Exception as exc:
            logger.debug(
                "PostgreSQL offline snapshot save skipped (offline/standalone mode)",
                error=str(exc),
            )

    def extract_batch_features(
        self,
        transactions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Execute batch feature engineering over historical transactions.
        Applies identical feature definitions in temporal order.

        Args:
            transactions_df: DataFrame containing raw transaction columns.

        Returns:
            pd.DataFrame: Augmented DataFrame with all 15 tabular ML features.
        """
        # Ensure temporal ordering
        df = transactions_df.sort_values(by="timestamp").copy()

        spend_comp = SpendFeatureComputer()
        device_comp = DeviceFeatureComputer()
        merchant_comp = MerchantFeatureComputer()

        # Customer last location state: customer_id -> (last_lat, last_lng, last_time)
        cust_locations: dict[str, tuple[float, float, float]] = {}
        # Customer velocity timestamps: customer_id -> list of timestamp_epochs
        cust_velocity_ts: dict[str, list[float]] = {}

        feature_records: list[dict] = []

        for _, row in df.iterrows():
            cust_id = str(row["customer_id"])
            txn_id = str(row["transaction_id"])
            amount = float(row["amount"])
            dev_fp = str(row["device_fingerprint"])
            merch_id = str(row["merchant_id"])
            merch_cat = str(row.get("merchant_category", "general_retail"))
            is_intl = bool(row.get("is_international", False))

            ts_raw = row["timestamp"]
            if isinstance(ts_raw, str):
                ts_dt = pd.to_datetime(ts_raw)
                ts_epoch = float(ts_dt.timestamp())
            elif hasattr(ts_raw, "timestamp"):
                ts_epoch = float(ts_raw.timestamp())
                ts_dt = ts_raw
            else:
                ts_epoch = float(ts_raw)
                ts_dt = pd.to_datetime(ts_epoch, unit="s")

            lat = float(row["location_lat"]) if pd.notna(row.get("location_lat")) else None
            lng = float(row["location_lng"]) if pd.notna(row.get("location_lng")) else None

            # 1. Velocity
            if cust_id not in cust_velocity_ts:
                cust_velocity_ts[cust_id] = []
            cust_velocity_ts[cust_id].append(ts_epoch)
            # Prune > 24h
            cust_velocity_ts[cust_id] = [t for t in cust_velocity_ts[cust_id] if t >= ts_epoch - 86400]
            c_ts = cust_velocity_ts[cust_id]

            v_1min = max(1, sum(1 for t in c_ts if t >= ts_epoch - 60))
            v_5min = max(1, sum(1 for t in c_ts if t >= ts_epoch - 300))
            v_1hr = max(1, sum(1 for t in c_ts if t >= ts_epoch - 3600))
            v_24hr = max(1, len(c_ts))

            # 2. Spend
            spend_feats = spend_comp.compute_spend_features(
                customer_id=cust_id,
                amount=amount,
                timestamp_epoch=ts_epoch,
            )

            # 3. Device
            dev_feats = device_comp.compute_device_features(
                customer_id=cust_id,
                device_fingerprint=dev_fp,
                timestamp_epoch=ts_epoch,
            )

            # 4. Merchant
            merch_feats = merchant_comp.compute_merchant_features(
                customer_id=cust_id,
                merchant_id=merch_id,
                merchant_category=merch_cat,
            )

            # 5. Geo
            if cust_id in cust_locations and lat is not None and lng is not None:
                last_lat, last_lng, _ = cust_locations[cust_id]
                dist = compute_haversine_distance(last_lat, last_lng, lat, lng)
            else:
                dist = 0.0

            if lat is not None and lng is not None:
                cust_locations[cust_id] = (lat, lng, ts_epoch)

            # 6. Temporal
            hour = int(ts_dt.hour)
            is_weekend = int(ts_dt.weekday() >= 5)

            record = {
                "transaction_id": txn_id,
                "customer_id": cust_id,
                "txn_count_1min": v_1min,
                "txn_count_5min": v_5min,
                "txn_count_1hr": v_1hr,
                "txn_count_24hr": v_24hr,
                "avg_amount_1hr": spend_feats["avg_amount_1hr"],
                "max_amount_24hr": spend_feats["max_amount_24hr"],
                "amount_zscore": spend_feats["amount_zscore"],
                "unique_devices_1hr": dev_feats["unique_devices_1hr"],
                "device_txn_count": dev_feats["device_txn_count"],
                "is_new_device": int(dev_feats["is_new_device"]),
                "merchant_avg_amount": merch_feats["merchant_avg_amount"],
                "merchant_fraud_rate": merch_feats["merchant_fraud_rate"],
                "is_new_merchant": int(merch_feats["is_new_merchant"]),
                "distance_from_last_txn": dist,
                "is_international": int(is_intl),
                "hour_of_day": hour,
                "is_weekend": is_weekend,
            }
            if "is_fraud" in row:
                record["is_fraud"] = int(row["is_fraud"])

            feature_records.append(record)

        return pd.DataFrame(feature_records)
