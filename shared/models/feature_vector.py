"""
SentinelForge — shared/models/feature_vector.py

PURPOSE:
    Defines Pydantic schemas for computed feature vectors, individual feature
    definition metadata, and offline feature store snapshots.

ARCHITECTURE POSITION:
    Layer 0 (Foundation) → Standardized representation exchanged between
    Feature Engine (Layer 2), Online Redis Store, Offline Postgres Store,
    and Real-Time LightGBM Scorer (Layer 3).

TRAINING-SERVING SKEW DEFENSE:
    Guarantees that whether features are extracted from streaming sliding windows
    or offline batch queries, they serialize into the exact same validated fields
    and numerical ranges before reaching the model.
"""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class FeatureDefinition(BaseModel):
    """
    Metadata specification for an individual feature.

    Attributes:
        name: Unique feature key name.
        window_seconds: Time window in seconds (e.g. 60, 300, 3600, 86400), or None for static.
        aggregation: Aggregation operator (count, mean, max, zscore, nunique, is_new, haversine).
        entity_key: Entity partition key (customer_id, device_fingerprint, merchant_id).
        source_field: Input transaction field to aggregate over.
        description: Human-readable documentation for feature catalog.
    """

    name: str = Field(..., description="Unique feature name")
    window_seconds: Optional[int] = Field(
        default=None, description="Sliding window duration in seconds"
    )
    aggregation: str = Field(..., description="Type of aggregation operator")
    entity_key: str = Field(..., description="Entity grouping identifier")
    source_field: Optional[str] = Field(
        default=None, description="Raw transaction attribute being aggregated"
    )
    description: Optional[str] = Field(
        default=None, description="Feature description for catalog"
    )


class FeatureVector(BaseModel):
    """
    Complete structured feature vector passed to LightGBM for fraud inference.

    Features cover 5 critical dimensions:
        1. Velocity: Burst frequency across 1min, 5min, 1hr, 24hr windows
        2. Spend Aggregates: Rolling averages, maximums, and z-score deviations
        3. Device Novelty: Hardware diversity and shared fingerprint frequencies
        4. Merchant Risk: Rolling spend and category fraud prevalence
        5. Geo-Temporal: Haversine distance from last known txn, international flag, hour of day
    """

    transaction_id: str = Field(..., description="Transaction identifier")
    customer_id: str = Field(..., description="Customer identifier")

    # Velocity Features
    txn_count_1min: int = Field(default=1, ge=0, description="Customer transactions in 1 minute")
    txn_count_5min: int = Field(default=1, ge=0, description="Customer transactions in 5 minutes")
    txn_count_1hr: int = Field(default=1, ge=0, description="Customer transactions in 1 hour")
    txn_count_24hr: int = Field(default=1, ge=0, description="Customer transactions in 24 hours")

    # Spend Features
    avg_amount_1hr: float = Field(default=0.0, ge=0.0, description="Average spend in 1 hour")
    max_amount_24hr: float = Field(default=0.0, ge=0.0, description="Max spend in 24 hours")
    amount_zscore: float = Field(
        default=0.0, description="Z-score deviation from customer historical mean"
    )

    # Device Features
    unique_devices_1hr: int = Field(
        default=1, ge=1, description="Unique devices seen for customer in 1 hour"
    )
    device_txn_count: int = Field(
        default=1, ge=0, description="Total transactions from this device fingerprint in 1 hour"
    )
    is_new_device: bool = Field(
        default=False, description="Whether device fingerprint is new for customer"
    )

    # Merchant Features
    merchant_avg_amount: float = Field(
        default=0.0, ge=0.0, description="Average transaction amount for merchant"
    )
    merchant_fraud_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Historical fraud rate for merchant"
    )
    is_new_merchant: bool = Field(
        default=False, description="Whether customer has never transacted with merchant"
    )

    # Geographic & Temporal Features
    distance_from_last_txn: float = Field(
        default=0.0, ge=0.0, description="Haversine distance (km) from previous transaction"
    )
    is_international: bool = Field(default=False, description="Cross-border indicator")
    hour_of_day: int = Field(default=12, ge=0, le=23, description="Hour of transaction (0-23)")
    is_weekend: bool = Field(default=False, description="Weekend indicator")

    # Snapshot metadata
    computed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Feature computation timestamp",
    )
    feature_version: str = Field(default="1.0.0", description="Feature schema definition version")

    def to_model_input_dict(self) -> dict[str, Any]:
        """
        Extract numeric and boolean features formatted for tabular ML scoring.

        Returns:
            dict[str, Any]: Pure feature map ready for LightGBM numpy array conversion.
        """
        exclude_keys = {"transaction_id", "customer_id", "computed_at", "feature_version"}
        return {
            k: (int(v) if isinstance(v, bool) else float(v))
            for k, v in self.model_dump().items()
            if k not in exclude_keys
        }


class FeatureSnapshotRecord(BaseModel):
    """
    Offline feature store record matching PostgreSQL `feature_snapshots` table.
    """

    id: UUID = Field(default_factory=uuid4, description="Primary database snapshot UUID")
    transaction_id: str = Field(..., description="Transaction identifier")
    features: dict[str, Any] = Field(..., description="Serialized feature dictionary")
    feature_version: str = Field(default="1.0.0", description="Feature version")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Insertion timestamp",
    )
