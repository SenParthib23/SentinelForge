"""
SentinelForge — Metrics Package

PURPOSE:
    Provides centralized Prometheus metrics and instrumentation utilities.
"""

from shared.metrics.prometheus import (
    DRIFT_DETECTED_TOTAL,
    FEATURE_FETCH_LATENCY_HISTOGRAM,
    INVESTIGATION_LATENCY_HISTOGRAM,
    MODEL_VERSION_INFO,
    SCORING_LATENCY_HISTOGRAM,
    SCORING_REQUESTS_TOTAL,
    track_feature_fetch_latency,
    track_investigation_latency,
    track_scoring_latency,
)

__all__ = [
    "SCORING_LATENCY_HISTOGRAM",
    "SCORING_REQUESTS_TOTAL",
    "FEATURE_FETCH_LATENCY_HISTOGRAM",
    "INVESTIGATION_LATENCY_HISTOGRAM",
    "MODEL_VERSION_INFO",
    "DRIFT_DETECTED_TOTAL",
    "track_scoring_latency",
    "track_feature_fetch_latency",
    "track_investigation_latency",
]
