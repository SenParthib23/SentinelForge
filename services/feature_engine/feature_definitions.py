"""
SentinelForge — services/feature_engine/feature_definitions.py

PURPOSE:
    THE SINGLE SOURCE OF TRUTH for all feature computations in SentinelForge.
    Both the real-time streaming pipeline (Faust/Redis) and the offline batch
    training pipeline (PostgreSQL/Parquet) import and execute the exact same
    mathematical definitions from this module.

ARCHITECTURE POSITION:
    Layer 2 (Streaming Feature Engine) → Shared by Online Scoring (Layer 3)
    and Offline ML Training (Layer 7).

TRAINING-SERVING SKEW DEFENSE:
    In real-time ML systems, training-serving skew is the most common silent bug.
    It occurs when a feature (e.g. "5-minute transaction count") is computed
    using SQL timestamp windowing during offline training, but streaming sliding
    windows with slightly different millisecond boundaries during live serving.
    By enforcing a single Python definition module containing parameter definitions,
    default values, and mathematical formulas, SentinelForge guarantees zero skew.

FEATURE TAXONOMY (15 Core Tabular Features):
    1. txn_count_1min: Rapid burst frequency in 60s window (Customer key)
    2. txn_count_5min: Medium burst frequency in 300s window (Customer key)
    3. txn_count_1hr: Hourly volume in 3600s window (Customer key)
    4. txn_count_24hr: Daily volume in 86400s window (Customer key)
    5. avg_amount_1hr: Rolling mean spend in 1hr window (Customer key)
    6. max_amount_24hr: Maximum single spend in 24hr window (Customer key)
    7. amount_zscore: Standard deviation units from customer historical mean
    8. unique_devices_1hr: Number of distinct device fingerprints in 1hr
    9. device_txn_count: Total transactions from this device fingerprint in 1hr
    10. is_new_device: Boolean flag if device fingerprint was never seen for customer
    11. merchant_avg_amount: Historical baseline mean transaction for merchant
    12. merchant_fraud_rate: Historical empirical fraud probability for merchant
    13. is_new_merchant: Boolean flag if customer has never transacted with merchant
    14. distance_from_last_txn: Haversine great-circle distance (km) from previous txn
    15. is_international: Cross-border binary indicator
    16. hour_of_day: Integer hour (0-23)
    17. is_weekend: Binary indicator for Saturday/Sunday
"""

from typing import Any, Callable, Dict, Final, List

FEATURE_VERSION: Final[str] = "1.0.0"

# Metadata Catalog defining every feature in the SentinelForge feature store
FEATURE_CATALOG: Final[Dict[str, Dict[str, Any]]] = {
    # 1. Velocity Features (Sliding Time Windows)
    "txn_count_1min": {
        "entity": "customer_id",
        "window_seconds": 60,
        "aggregation": "count",
        "default": 1,
        "dtype": "int",
        "description": "Number of transactions by customer in the last 60 seconds",
    },
    "txn_count_5min": {
        "entity": "customer_id",
        "window_seconds": 300,
        "aggregation": "count",
        "default": 1,
        "dtype": "int",
        "description": "Number of transactions by customer in the last 5 minutes",
    },
    "txn_count_1hr": {
        "entity": "customer_id",
        "window_seconds": 3600,
        "aggregation": "count",
        "default": 1,
        "dtype": "int",
        "description": "Number of transactions by customer in the last 1 hour",
    },
    "txn_count_24hr": {
        "entity": "customer_id",
        "window_seconds": 86400,
        "aggregation": "count",
        "default": 1,
        "dtype": "int",
        "description": "Number of transactions by customer in the last 24 hours",
    },
    # 2. Spend Aggregate Features
    "avg_amount_1hr": {
        "entity": "customer_id",
        "window_seconds": 3600,
        "aggregation": "mean",
        "source_field": "amount",
        "default": 0.0,
        "dtype": "float",
        "description": "Rolling average transaction amount for customer in last 1 hour",
    },
    "max_amount_24hr": {
        "entity": "customer_id",
        "window_seconds": 86400,
        "aggregation": "max",
        "source_field": "amount",
        "default": 0.0,
        "dtype": "float",
        "description": "Maximum transaction amount by customer in last 24 hours",
    },
    "amount_zscore": {
        "entity": "customer_id",
        "window_seconds": None,
        "aggregation": "zscore",
        "source_field": "amount",
        "default": 0.0,
        "dtype": "float",
        "description": "Z-score deviation: (amount - customer_mean) / customer_std",
    },
    # 3. Device & Novelty Features
    "unique_devices_1hr": {
        "entity": "customer_id",
        "window_seconds": 3600,
        "aggregation": "nunique",
        "source_field": "device_fingerprint",
        "default": 1,
        "dtype": "int",
        "description": "Distinct device fingerprints used by customer in last 1 hour",
    },
    "device_txn_count": {
        "entity": "device_fingerprint",
        "window_seconds": 3600,
        "aggregation": "count",
        "default": 1,
        "dtype": "int",
        "description": "Total transactions across all accounts from this device in 1 hour",
    },
    "is_new_device": {
        "entity": "customer_id",
        "window_seconds": 604800,  # 7 days lookback
        "aggregation": "is_new",
        "source_field": "device_fingerprint",
        "default": False,
        "dtype": "bool",
        "description": "True if this device fingerprint has not been seen for customer in 7 days",
    },
    # 4. Merchant Risk Features
    "merchant_avg_amount": {
        "entity": "merchant_id",
        "window_seconds": 86400,
        "aggregation": "mean",
        "source_field": "amount",
        "default": 500.0,
        "dtype": "float",
        "description": "Historical average transaction amount processed by merchant",
    },
    "merchant_fraud_rate": {
        "entity": "merchant_id",
        "window_seconds": None,
        "aggregation": "ratio",
        "default": 0.01,
        "dtype": "float",
        "description": "Historical empirical fraud probability for merchant",
    },
    "is_new_merchant": {
        "entity": "customer_id",
        "window_seconds": 604800,
        "aggregation": "is_new",
        "source_field": "merchant_id",
        "default": False,
        "dtype": "bool",
        "description": "True if customer has never transacted with this merchant",
    },
    # 5. Geographic & Temporal Features
    "distance_from_last_txn": {
        "entity": "customer_id",
        "window_seconds": None,
        "aggregation": "haversine",
        "default": 0.0,
        "dtype": "float",
        "description": "Haversine great-circle distance (km) from customer's previous transaction",
    },
    "is_international": {
        "entity": None,
        "window_seconds": None,
        "aggregation": "passthrough",
        "source_field": "is_international",
        "default": False,
        "dtype": "bool",
        "description": "Whether transaction is cross-border",
    },
    "hour_of_day": {
        "entity": None,
        "window_seconds": None,
        "aggregation": "extract_hour",
        "source_field": "timestamp",
        "default": 12,
        "dtype": "int",
        "description": "Hour of day (0-23) in UTC",
    },
    "is_weekend": {
        "entity": None,
        "window_seconds": None,
        "aggregation": "extract_weekend",
        "source_field": "timestamp",
        "default": False,
        "dtype": "bool",
        "description": "True if transaction occurred on Saturday or Sunday",
    },
}

ORDERED_FEATURE_NAMES: Final[List[str]] = list(FEATURE_CATALOG.keys())
