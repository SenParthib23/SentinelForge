"""
SentinelForge — services/scoring/canary_router.py

PURPOSE:
    Dynamic traffic splitter for Canary Deployments and Shadow Scoring.
    Partitions real-time transaction authorization traffic between the active
    Production Champion and candidate Canary models using deterministic customer hashing.

ARCHITECTURE POSITION:
    Layer 10 (Model Deployment & Serving) → Traffic Routing & Canary Gate.

INTERVIEW TALKING POINT:
    "We don't use simple random routing for canary deployments because a cardholder making
    two rapid transactions could be scored by two different models with different thresholds.
    We hash customer_id modulo 100 to guarantee deterministic sticky routing, while supporting
    shadow mode where the candidate model scores asynchronously without impacting payment auth."
"""

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from shared.logging.logger import get_logger

logger = get_logger("scoring.canary_router")


@dataclass
class CanaryConfig:
    """Configuration for canary deployment traffic routing."""
    champion_version: str = "baseline_v1"
    canary_version: Optional[str] = None
    canary_percentage: float = 0.0          # e.g., 10.0 for 10% canary traffic
    is_shadow_mode: bool = False            # In shadow mode, candidate runs async without auth impact
    is_active: bool = False                 # Whether canary routing is active


class CanaryRouter:
    """
    Manages deterministic traffic distribution between Champion and Canary models.
    """

    def __init__(self, config: Optional[CanaryConfig] = None) -> None:
        """
        Initialize CanaryRouter.

        Args:
            config: Initial canary configuration.
        """
        self.config = config or CanaryConfig()

    def route_transaction(self, customer_id: str) -> Tuple[str, bool]:
        """
        Determine which model version should score this transaction.

        Args:
            customer_id: Customer identifier for sticky deterministic hashing.

        Returns:
            Tuple[str, bool]: (Selected model version, Is canary routing).
        """
        if not self.config.is_active or not self.config.canary_version or self.config.canary_percentage <= 0.0:
            return self.config.champion_version, False

        # If in shadow mode, primary routing always goes to champion
        if self.config.is_shadow_mode:
            return self.config.champion_version, False

        # Deterministic MD5 hash modulo 100 for consistent cardholder stickiness
        h = int(hashlib.md5(customer_id.encode("utf-8")).hexdigest(), 16) % 100
        if h < self.config.canary_percentage:
            return self.config.canary_version, True

        return self.config.champion_version, False

    def should_shadow_score(self) -> bool:
        """
        Check if shadow execution should run in parallel for this transaction.

        Returns:
            bool: True if shadow candidate should evaluate asynchronously.
        """
        return (
            self.config.is_active
            and self.config.is_shadow_mode
            and self.config.canary_version is not None
        )

    def update_split(
        self,
        canary_percentage: float,
        canary_version: Optional[str] = None,
        shadow_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Update dynamic traffic split ratio.

        Args:
            canary_percentage: Percentage of traffic routed to canary [0.0, 100.0].
            canary_version: Target canary model tag.
            shadow_mode: Whether candidate runs in shadow evaluation mode.

        Returns:
            Dict[str, Any]: Updated configuration snapshot.
        """
        if not (0.0 <= canary_percentage <= 100.0):
            raise ValueError(f"Canary percentage must be between 0.0 and 100.0, got {canary_percentage}")

        if canary_version:
            self.config.canary_version = canary_version

        self.config.canary_percentage = canary_percentage
        self.config.is_shadow_mode = shadow_mode
        self.config.is_active = canary_percentage > 0.0 or shadow_mode

        logger.info(
            "Updated canary deployment traffic split",
            champion=self.config.champion_version,
            canary=self.config.canary_version,
            canary_percentage=self.config.canary_percentage,
            shadow_mode=self.config.is_shadow_mode,
            is_active=self.config.is_active,
        )

        return {
            "champion_version": self.config.champion_version,
            "canary_version": self.config.canary_version,
            "canary_percentage": self.config.canary_percentage,
            "shadow_mode": self.config.is_shadow_mode,
            "is_active": self.config.is_active,
        }

    def reset_to_champion(self) -> None:
        """
        Instantly reset all traffic to 100% Champion (used during rollbacks).
        """
        self.config.canary_percentage = 0.0
        self.config.is_active = False
        self.config.is_shadow_mode = False
        logger.warning(
            "Canary traffic instantly terminated; 100% traffic restored to champion",
            champion=self.config.champion_version,
        )


_global_canary_router: Optional[CanaryRouter] = None


def get_canary_router() -> CanaryRouter:
    """Retrieve shared CanaryRouter singleton instance."""
    global _global_canary_router
    if _global_canary_router is None:
        _global_canary_router = CanaryRouter()
    return _global_canary_router
