"""
SentinelForge — Transaction Simulator Package

PURPOSE:
    Simulates high-throughput financial transactions with realistic normal spending
    behaviors and five explicit fraud typologies.
"""

from services.transaction_simulator.simulator import (
    CustomerProfile,
    FraudPatternType,
    MerchantProfile,
    TransactionSimulator,
)

__all__ = [
    "FraudPatternType",
    "CustomerProfile",
    "MerchantProfile",
    "TransactionSimulator",
]
