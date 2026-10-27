"""
SentinelForge — ml/monitoring/drift_detector.py

PURPOSE:
    Continuous Data and Concept Drift Detection engine.
    Computes Population Stability Index (PSI) and Wasserstein Distance across all 17
    behavioral features between historical reference baselines and streaming inference windows.

ARCHITECTURE POSITION:
    Layer 11 (Monitoring & Observability) → Continuous Drift Detection.

INTERVIEW TALKING POINT:
    "Financial fraud patterns mutate rapidly due to adversarial adaptation by crime syndicates.
    DriftDetector monitors all 17 features in real time using Population Stability Index (PSI).
    A PSI < 0.10 indicates stability; 0.10 <= PSI < 0.25 flags moderate behavioral change;
    and PSI >= 0.25 triggers an automated retraining recommendation before model accuracy degrades."
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from services.scoring.scorer import FEATURE_COLUMN_ORDER
from shared.logging.logger import get_logger

logger = get_logger("ml.monitoring.drift")


@dataclass
class FeatureDriftResult:
    """Drift metrics for a single feature."""
    feature_name: str
    psi: float
    wasserstein_dist: float
    is_drifted: bool
    severity: str  # 'NONE', 'MODERATE', 'SEVERE'


@dataclass
class DriftReport:
    """Consolidated drift report across all monitored features."""
    timestamp: str
    overall_drift_detected: bool
    mean_psi: float
    features_monitored: int
    drifted_features: List[str]
    feature_metrics: Dict[str, FeatureDriftResult] = field(default_factory=dict)


class DriftDetector:
    """
    Computes statistical divergence (PSI and Wasserstein distance) across feature distributions.
    """

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        psi_threshold_warning: float = 0.10,
        psi_threshold_alert: float = 0.25,
        num_bins: int = 10,
    ) -> None:
        """
        Initialize DriftDetector with reference baseline.

        Args:
            reference_data: Baseline training set feature DataFrame.
            psi_threshold_warning: Threshold for moderate drift warning (0.10).
            psi_threshold_alert: Threshold for severe drift alert (0.25).
            num_bins: Number of quantile bins for PSI computation.
        """
        self.ref_data = reference_data
        self.psi_warn = psi_threshold_warning
        self.psi_alert = psi_threshold_alert
        self.num_bins = num_bins

    def compute_feature_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """
        Compute Population Stability Index (PSI) between reference and current samples.

        Args:
            reference: 1D array of reference baseline feature values.
            current: 1D array of current inference feature values.

        Returns:
            float: Population Stability Index.
        """
        if len(reference) == 0 or len(current) == 0:
            return 0.0

        # Compute bin edges from reference distribution
        percentiles = np.linspace(0, 100, self.num_bins + 1)
        bin_edges = np.percentile(reference, percentiles)
        bin_edges = np.unique(bin_edges)  # Handle constant or low-cardinality features

        if len(bin_edges) < 2:
            return 0.0

        # Set outer boundaries to infinity to capture out-of-range samples
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Calculate frequency counts in each bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert to proportions with small epsilon to prevent division by zero / log(0)
        eps = 1e-4
        ref_pct = (ref_counts + eps) / (len(reference) + eps * len(ref_counts))
        cur_pct = (cur_counts + eps) / (len(current) + eps * len(cur_counts))

        # PSI formula: sum((Actual - Expected) * ln(Actual / Expected))
        psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(np.clip(psi_val, 0.0, 10.0))

    def compute_drift_report(
        self,
        current_data: pd.DataFrame,
        reference_data: Optional[pd.DataFrame] = None,
    ) -> DriftReport:
        """
        Compute drift across all 17 features.

        Args:
            current_data: Current inference window DataFrame.
            reference_data: Optional override of reference DataFrame.

        Returns:
            DriftReport: Consolidated drift health scorecard.
        """
        ref_df = reference_data if reference_data is not None else self.ref_data

        if ref_df is None:
            logger.warning("No reference dataset supplied; generating synthetic reference for initialization")
            ref_df = current_data.copy()

        features_to_check = [col for col in FEATURE_COLUMN_ORDER if col in current_data.columns and col in ref_df.columns]
        extra_numeric = [
            col for col in current_data.columns
            if col in ref_df.columns
            and col not in features_to_check
            and pd.api.types.is_numeric_dtype(current_data[col])
        ]
        features_to_check.extend(extra_numeric)

        feature_metrics: Dict[str, FeatureDriftResult] = {}
        drifted_features: List[str] = []
        psi_values: List[float] = []

        for feat in features_to_check:
            ref_col = ref_df[feat].dropna().values.astype(float)
            cur_col = current_data[feat].dropna().values.astype(float)

            if len(ref_col) < 5 or len(cur_col) < 5:
                continue

            psi = self.compute_feature_psi(ref_col, cur_col)
            w_dist = float(wasserstein_distance(ref_col, cur_col))
            psi_values.append(psi)

            if psi >= self.psi_alert:
                severity = "SEVERE"
                is_drifted = True
                drifted_features.append(feat)
            elif psi >= self.psi_warn:
                severity = "MODERATE"
                is_drifted = True
                drifted_features.append(feat)
            else:
                severity = "NONE"
                is_drifted = False

            feature_metrics[feat] = FeatureDriftResult(
                feature_name=feat,
                psi=round(psi, 4),
                wasserstein_dist=round(w_dist, 4),
                is_drifted=is_drifted,
                severity=severity,
            )

        mean_psi = float(np.mean(psi_values)) if psi_values else 0.0
        overall_drift = len(drifted_features) > 0 or mean_psi >= self.psi_warn

        report = DriftReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_drift_detected=overall_drift,
            mean_psi=round(mean_psi, 4),
            features_monitored=len(features_to_check),
            drifted_features=drifted_features,
            feature_metrics=feature_metrics,
        )

        logger.info(
            "Drift analysis complete",
            overall_drift=overall_drift,
            mean_psi=round(mean_psi, 4),
            drifted_count=len(drifted_features),
            drifted_features=drifted_features,
        )
        return report
