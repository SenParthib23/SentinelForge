"""
SentinelForge — Feature Engine Package

PURPOSE:
    Provides real-time streaming feature computation, Redis online store persistence,
    and PostgreSQL/Parquet offline store feature transformation.
"""

from services.feature_engine.device_features import DeviceFeatureComputer
from services.feature_engine.faust_app import StreamingFeaturePipeline
from services.feature_engine.feature_definitions import (
    FEATURE_CATALOG,
    FEATURE_VERSION,
    ORDERED_FEATURE_NAMES,
)
from services.feature_engine.geo_features import compute_haversine_distance
from services.feature_engine.graph_features import GraphFeatureComputer
from services.feature_engine.merchant_features import MerchantFeatureComputer
from services.feature_engine.offline_store import OfflineFeatureStore
from services.feature_engine.online_store import OnlineFeatureStore
from services.feature_engine.spend_features import SpendFeatureComputer
from services.feature_engine.velocity_features import VelocityFeatureComputer

__all__ = [
    "FEATURE_VERSION",
    "FEATURE_CATALOG",
    "ORDERED_FEATURE_NAMES",
    "compute_haversine_distance",
    "VelocityFeatureComputer",
    "SpendFeatureComputer",
    "DeviceFeatureComputer",
    "MerchantFeatureComputer",
    "GraphFeatureComputer",
    "OnlineFeatureStore",
    "OfflineFeatureStore",
    "StreamingFeaturePipeline",
]
