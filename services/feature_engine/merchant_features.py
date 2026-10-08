"""
SentinelForge — services/feature_engine/merchant_features.py

PURPOSE:
    Computes merchant baseline risk profiles, historical merchant fraud rates,
    and customer-merchant interaction novelty.

ARCHITECTURE POSITION:
    Layer 2 (Feature Engine) → Merchant Risk Subsystem.

FEATURES COMPUTED:
    - merchant_avg_amount: Baseline average charge size for merchant.
    - merchant_fraud_rate: Historical fraud likelihood for merchant category.
    - is_new_merchant: True if customer has never transacted with this merchant.
"""

from collections import defaultdict
from typing import Optional

# Base risk rates by category
DEFAULT_CATEGORY_RISK: dict[str, tuple[float, float]] = {
    "grocery": (450.0, 0.005),
    "fuel": (1500.0, 0.010),
    "dining": (850.0, 0.012),
    "electronics": (12000.0, 0.045),
    "pharmacy": (600.0, 0.008),
    "travel": (18000.0, 0.035),
    "crypto_exchange": (25000.0, 0.120),
    "luxury_jewelry": (45000.0, 0.095),
    "gambling": (8000.0, 0.150),
    "online_services": (999.0, 0.020),
}


class MerchantFeatureComputer:
    """
    Evaluates merchant risk baselines and customer merchant affinity.
    """

    def __init__(self) -> None:
        """Initialize merchant and customer interaction state."""
        # customer_id -> set of known merchant_ids
        self._customer_known_merchants: dict[str, set[str]] = defaultdict(set)
        # merchant_id -> (avg_amount, fraud_rate)
        self._merchant_profiles: dict[str, tuple[float, float]] = {}

    def register_merchant_profile(
        self, merchant_id: str, avg_amount: float, fraud_rate: float
    ) -> None:
        """Register specific merchant historical profile."""
        self._merchant_profiles[merchant_id] = (avg_amount, fraud_rate)

    def register_known_merchant(self, customer_id: str, merchant_id: str) -> None:
        """Pre-seed customer's historical visited merchant."""
        self._customer_known_merchants[customer_id].add(merchant_id)

    def compute_merchant_features(
        self,
        customer_id: str,
        merchant_id: str,
        merchant_category: Optional[str] = None,
    ) -> dict[str, any]:
        """
        Compute merchant risk attributes and novelty for customer.

        Args:
            customer_id: Customer ID.
            merchant_id: Merchant ID.
            merchant_category: Optional category name.

        Returns:
            dict[str, any]: Computed merchant features:
                - merchant_avg_amount: float
                - merchant_fraud_rate: float
                - is_new_merchant: bool
        """
        # 1. Determine novelty
        is_new = merchant_id not in self._customer_known_merchants[customer_id]
        self._customer_known_merchants[customer_id].add(merchant_id)

        # 2. Determine profile
        if merchant_id in self._merchant_profiles:
            avg_amt, fraud_rate = self._merchant_profiles[merchant_id]
        elif merchant_category and merchant_category in DEFAULT_CATEGORY_RISK:
            avg_amt, fraud_rate = DEFAULT_CATEGORY_RISK[merchant_category]
        else:
            avg_amt, fraud_rate = 1000.0, 0.015

        return {
            "merchant_avg_amount": round(float(avg_amt), 2),
            "merchant_fraud_rate": round(float(fraud_rate), 4),
            "is_new_merchant": is_new,
        }
