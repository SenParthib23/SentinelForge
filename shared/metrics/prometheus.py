"""
SentinelForge — shared/metrics/prometheus.py

PURPOSE:
    Provides centralized Prometheus metrics and latency histograms for the
    real-time scoring path (<150ms SLA) and asynchronous LLM investigation assist.

ARCHITECTURE POSITION:
    Layer 0 (Foundation) → Scraped by Prometheus container via `/api/v1/health/metrics`
    and displayed on Grafana operations & lifecycle dashboards.

METRIC DEFINITIONS:
    - scoring_latency_seconds: Histogram tracking LightGBM inference latency (sub-10ms SLA).
    - feature_fetch_latency_seconds: Histogram tracking Redis online feature retrieval latency.
    - scoring_requests_total: Counter of scored transactions partitioned by risk label and version.
    - investigation_latency_seconds: Histogram tracking LLM case narrative generation (3-5s SLA).
    - model_version_status: Gauge reporting active deployment state per model.
    - drift_detected_total: Counter tracking feature and prediction distribution drifts.

INTERVIEW TALKING POINT:
    "We configured custom non-linear histogram buckets for scoring latency: 1ms, 5ms,
    10ms, 25ms, 50ms, 100ms, and 150ms. Since the transaction authorization SLA is hard-capped
    at 150ms and LightGBM should score under 10ms, default exponential Prometheus buckets
    would be too coarse to detect p99 latency regressions before our payment gateway times out."
"""

import time
from contextlib import contextmanager
from typing import Generator

from prometheus_client import Counter, Gauge, Histogram

# Scoring Latency Histogram (Custom buckets matching <150ms auth path SLA)
SCORING_LATENCY_HISTOGRAM = Histogram(
    "sentinelforge_scoring_latency_seconds",
    "Real-time LightGBM inference and scoring decision latency in seconds",
    ["model_version"],
    buckets=(0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.150, 0.250, 0.500),
)

# Feature Fetch Latency Histogram (Redis sub-millisecond to low millisecond target)
FEATURE_FETCH_LATENCY_HISTOGRAM = Histogram(
    "sentinelforge_feature_fetch_latency_seconds",
    "Time taken to fetch online feature vector from Redis in seconds",
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.010, 0.025, 0.050),
)

# Scoring Request Counter
SCORING_REQUESTS_TOTAL = Counter(
    "sentinelforge_scoring_requests_total",
    "Total transaction scoring requests processed by risk label and model version",
    ["risk_label", "model_version"],
)

# Investigation Latency Histogram (Off-critical path LLM generation)
INVESTIGATION_LATENCY_HISTOGRAM = Histogram(
    "sentinelforge_investigation_latency_seconds",
    "Time taken for LLM to assemble narrative and verify evidence in seconds",
    ["model_version"],
    buckets=(0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0),
)

# Active Model Deployment State (1 = Active, 0 = Inactive)
MODEL_VERSION_INFO = Gauge(
    "sentinelforge_model_version_status",
    "Current status of models in registry (candidate, canary, production, retired)",
    ["model_name", "version", "status"],
)

# Drift Detection Counter
DRIFT_DETECTED_TOTAL = Counter(
    "sentinelforge_drift_detected_total",
    "Number of statistically significant drift events detected by Evidently AI",
    ["model_name", "drift_type"],
)


@contextmanager
def track_scoring_latency(model_version: str) -> Generator[None, None, None]:
    """
    Context manager to observe scoring duration and record to Prometheus histogram.

    Args:
        model_version: Identifier of the active model performing inference.

    Yields:
        None: Yields control during inference execution.
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        SCORING_LATENCY_HISTOGRAM.labels(model_version=model_version).observe(elapsed)


@contextmanager
def track_feature_fetch_latency() -> Generator[None, None, None]:
    """
    Context manager to observe Redis online feature fetch duration.

    Yields:
        None: Yields control during Redis I/O.
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        FEATURE_FETCH_LATENCY_HISTOGRAM.observe(elapsed)


@contextmanager
def track_investigation_latency(model_version: str) -> Generator[None, None, None]:
    """
    Context manager to observe LLM narrative generation latency.

    Args:
        model_version: LLM identifier / adapter version generating narrative.

    Yields:
        None: Yields control during Groq API execution.
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        INVESTIGATION_LATENCY_HISTOGRAM.labels(model_version=model_version).observe(elapsed)
