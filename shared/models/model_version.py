"""
SentinelForge — shared/models/model_version.py

PURPOSE:
    Defines Pydantic schemas for model registry tracking, evaluation benchmarks,
    regression reports, and canary rollout observability.

ARCHITECTURE POSITION:
    Layer 0 (Foundation) → Consumed by Model Registry (Layer 9), Eval Harness,
    Promotion Pipeline, and FastAPI Model Management Router.

EVAL-GATED PROMOTION PATTERN:
    Models must satisfy two distinct gates before canary deployment:
    1. Benchmark Eval: Absolute performance thresholds on clean test datasets.
    2. Regression Eval: Head-to-head performance on historical traffic slices vs.
       the current production champion to catch subtle edge-case degradations.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ModelStatus(str, Enum):
    """Model deployment lifecycle state."""

    CANDIDATE = "candidate"      # Newly trained model awaiting evaluation
    CANARY = "canary"            # Receiving partial traffic (e.g. 10%)
    PRODUCTION = "production"    # Active primary champion model
    RETIRED = "retired"          # Decommissioned model


class CanaryMetrics(BaseModel):
    """
    Live performance metrics collected during canary deployment window.
    """

    traffic_percentage: float = Field(default=10.0, description="Percentage of traffic routed")
    transactions_scored: int = Field(default=0, description="Total scored transactions")
    avg_latency_ms: float = Field(default=0.0, description="Mean inference latency")
    p95_latency_ms: float = Field(default=0.0, description="95th percentile latency")
    p99_latency_ms: float = Field(default=0.0, description="99th percentile latency")
    flag_rate: float = Field(default=0.0, description="Percentage of transactions flagged")
    error_count: int = Field(default=0, description="Inference error count")


class EvalReport(BaseModel):
    """
    Comprehensive evaluation report comparing candidate against production baseline.
    """

    model_version: str = Field(..., description="Candidate model version evaluated")
    baseline_version: Optional[str] = Field(
        default=None, description="Current production model version compared against"
    )
    auc_roc: float = Field(..., ge=0.0, le=1.0, description="Area under ROC curve")
    precision: float = Field(..., ge=0.0, le=1.0, description="Precision score")
    recall: float = Field(..., ge=0.0, le=1.0, description="Recall score")
    f1_score: float = Field(..., ge=0.0, le=1.0, description="F1 score")
    precision_at_90_recall: Optional[float] = Field(
        default=None, description="Precision when recall is pinned to 90%"
    )
    p99_latency_ms: float = Field(..., description="Simulated p99 inference latency")
    passed_benchmark: bool = Field(..., description="Whether absolute criteria passed")
    passed_regression: bool = Field(..., description="Whether head-to-head regression passed")
    regression_summary: dict[str, Any] = Field(
        default_factory=dict, description="Comparison breakdown against current production"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report generation timestamp",
    )


class ModelVersion(BaseModel):
    """
    Complete model version record matching PostgreSQL `model_versions` table.
    """

    id: UUID = Field(default_factory=uuid4, description="Model record UUID")
    model_name: str = Field(..., description="Name of model family (e.g. lightgbm_fraud)")
    version: str = Field(..., description="Semantic version string (e.g. v1, v2)")
    mlflow_run_id: Optional[str] = Field(default=None, description="Associated MLflow run ID")
    base_model: Optional[str] = Field(default=None, description="Base foundation model if adapter")
    dataset_hash: Optional[str] = Field(default=None, description="SHA256 hash of training data")
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )
    eval_metrics: dict[str, Any] = Field(
        default_factory=dict, description="Benchmark and regression metrics"
    )
    status: ModelStatus = Field(default=ModelStatus.CANDIDATE, description="Current status")
    canary_start: Optional[datetime] = Field(default=None, description="Canary rollout start time")
    canary_end: Optional[datetime] = Field(default=None, description="Canary rollout end time")
    canary_metrics: Optional[dict[str, Any]] = Field(
        default=None, description="Live metrics observed in canary"
    )
    promoted_at: Optional[datetime] = Field(
        default=None, description="Production promotion timestamp"
    )
    retired_at: Optional[datetime] = Field(
        default=None, description="Retirement timestamp"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Model registration timestamp",
    )
