r"""
SentinelForge — services/feature_engine/spend_features.py

PURPOSE:
    Computes rolling spend aggregates (mean, max) and statistical z-score deviations
    from the customer's historical spending profile.

ARCHITECTURE POSITION:
    Layer 2 (Feature Engine) → Spend Feature Subsystem.

MATHEMATICAL FORMULATION:
    For a transaction amount $x$ given customer historical baseline $(\mu, \sigma)$:
    $$Z = \frac{x - \mu}{\max(\sigma, 10.0)}$$
    A z-score exceeding $+3.0$ indicates an amount in the 99.7th percentile of the
    customer's normal spending distribution, presenting a strong anomaly signal.

INTERVIEW TALKING POINT:
    "We clip the divisor in the z-score calculation to a minimum standard deviation
    threshold (10.0 INR) to prevent division by zero or extreme sensitivity on
    customers who consistently make identical low-denomination recurring payments."
"""

from collections import defaultdict
from typing import Optional

from shared.logging.logger import get_logger

logger = get_logger("features.spend")


class SpendFeatureComputer:
    """
    Computes rolling spend aggregates and statistical anomaly z-scores.
    """

    def __init__(self) -> None:
        """Initialize in-memory spend cache per customer."""
        # customer_id -> list of (amount, timestamp_epoch)
        self._customer_spends: dict[str, list[tuple[float, float]]] = defaultdict(list)
        # customer_id -> (baseline_mean, baseline_std)
        self._customer_baselines: dict[str, tuple[float, float]] = {}

    def register_baseline(
        self, customer_id: str, mean_amount: float, std_amount: float
    ) -> None:
        """
        Register known customer baseline profile from historical offline feature store.

        Args:
            customer_id: Customer ID.
            mean_amount: Customer historical average transaction amount.
            std_amount: Customer historical standard deviation of transaction amount.
        """
        self._customer_baselines[customer_id] = (
            max(10.0, mean_amount),
            max(5.0, std_amount),
        )

    def compute_spend_features(
        self,
        customer_id: str,
        amount: float,
        timestamp_epoch: float,
        baseline_mean: Optional[float] = None,
        baseline_std: Optional[float] = None,
    ) -> dict[str, float]:
        """
        Record transaction spend and compute rolling 1hr average, 24hr max, and z-score.

        Args:
            customer_id: Customer identifier.
            amount: Transaction monetary amount.
            timestamp_epoch: Transaction Unix timestamp.
            baseline_mean: Optional override baseline mean.
            baseline_std: Optional override baseline standard deviation.

        Returns:
            dict[str, float]: Computed spend features:
                - avg_amount_1hr
                - max_amount_24hr
                - amount_zscore
        """
        # Append current transaction
        self._customer_spends[customer_id].append((amount, timestamp_epoch))

        # Prune transactions older than 24 hours (86400s)
        cutoff_24hr = timestamp_epoch - 86400
        cutoff_1hr = timestamp_epoch - 3600

        self._customer_spends[customer_id] = [
            (amt, ts)
            for amt, ts in self._customer_spends[customer_id]
            if ts >= cutoff_24hr
        ]

        recent_spends_1hr = [
            amt for amt, ts in self._customer_spends[customer_id] if ts >= cutoff_1hr
        ]
        recent_spends_24hr = [amt for amt, ts in self._customer_spends[customer_id]]

        avg_1hr = (
            sum(recent_spends_1hr) / len(recent_spends_1hr)
            if recent_spends_1hr
            else amount
        )
        max_24hr = max(recent_spends_24hr) if recent_spends_24hr else amount

        # Determine baseline for z-score
        if baseline_mean is not None and baseline_std is not None:
            b_mean, b_std = baseline_mean, baseline_std
        elif customer_id in self._customer_baselines:
            b_mean, b_std = self._customer_baselines[customer_id]
        else:
            # Cold-start default: use 1hr or overall recent mean
            b_mean = avg_1hr
            b_std = max(10.0, avg_1hr * 0.5)

        zscore = (amount - b_mean) / max(10.0, b_std)

        return {
            "avg_amount_1hr": round(avg_1hr, 2),
            "max_amount_24hr": round(max_24hr, 2),
            "amount_zscore": round(zscore, 3),
        }
