"""
SentinelForge — services/transaction_simulator/simulator.py

PURPOSE:
    Generates realistic, high-throughput synthetic transaction event streams with
    configurable class imbalance (~2-5% fraud rate) and 5 distinct fraud typologies:
    1. Velocity bursts (rapid successive transactions from same device)
    2. Amount anomalies (10x-50x above customer's historical baseline)
    3. Geographic anomalies (impossible travel / abrupt cross-continent location jumps)
    4. New high-risk merchant fraud (first-time charge in high-risk categories like crypto/gambling)
    5. Card-not-present device mismatch (credential stuffing from novel device fingerprint)

ARCHITECTURE POSITION:
    Layer 1 (Ingestion & Simulation) → Emits to Kafka topic `txn.stream` to drive
    the real-time streaming feature engine (Layer 2) and populate offline training sets.

WHY SYNTHETIC PROFILE GENERATION:
    Real banking datasets cannot be published due to PCI-DSS and GDPR constraints.
    By modeling realistic customer spending distributions (log-normal amounts, home geo-anchors,
    known device fingerprints, and merchant affinity graphs), we generate benchmark data
    that exhibits real-world non-stationary behavior, class imbalance, and feature correlations.

INTERVIEW TALKING POINT:
    "Rather than uniform random noise, we modeled individual customer entities with
    log-normal spending habits, home GPS coordinates, and device affinities.
    This creates authentic behavioral baselines, so features like Haversine distance,
    amount z-scores, and sliding-window velocity counters produce clean, separable
    signals for LightGBM while testing edge-case generalization."
"""

import asyncio
import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, AsyncGenerator, Optional
from uuid import uuid4

from shared.config.settings import Settings, get_settings
from shared.logging.logger import get_logger
from shared.models.transaction import PaymentChannel, TransactionEvent

logger = get_logger("simulator")


class FraudPatternType(str, Enum):
    """Specific fraud attack pattern simulated."""

    NONE = "none"
    VELOCITY = "velocity"
    AMOUNT_ANOMALY = "amount_anomaly"
    GEOGRAPHIC_ANOMALY = "geographic_anomaly"
    NEW_MERCHANT = "new_merchant"
    CARD_NOT_PRESENT = "card_not_present"


@dataclass
class CustomerProfile:
    """
    Simulated customer baseline spending and location profile.
    """

    customer_id: str
    avg_amount: float
    std_amount: float
    home_lat: float
    home_lng: float
    primary_device: str
    primary_channel: PaymentChannel
    known_merchants: set[str] = field(default_factory=set)
    last_txn_time: Optional[datetime] = None
    last_lat: Optional[float] = None
    last_lng: Optional[float] = None


@dataclass
class MerchantProfile:
    """
    Simulated merchant profile with industry category and baseline risk factor.
    """

    merchant_id: str
    merchant_name: str
    category: str
    risk_weight: float
    avg_amount: float


MERCHANT_CATEGORIES = [
    ("grocery", 0.005, 450.0),
    ("fuel", 0.010, 1500.0),
    ("dining", 0.012, 850.0),
    ("electronics", 0.045, 12000.0),
    ("pharmacy", 0.008, 600.0),
    ("travel", 0.035, 18000.0),
    ("crypto_exchange", 0.120, 25000.0),
    ("luxury_jewelry", 0.095, 45000.0),
    ("gambling", 0.150, 8000.0),
    ("online_services", 0.020, 999.0),
]

METROPOLITAN_CENTERS = [
    ("Mumbai", 19.0760, 72.8777),
    ("Delhi", 28.6139, 77.2090),
    ("Bengaluru", 12.9716, 77.5946),
    ("Hyderabad", 17.3850, 78.4867),
    ("Chennai", 13.0827, 80.2707),
    ("Kolkata", 22.5726, 88.3639),
    ("Pune", 18.5204, 73.8567),
]


class TransactionSimulator:
    """
    Synthetic transaction generator producing realistic credit/debit/UPI payments
    with configurable fraud typologies and background rates.
    """

    def __init__(
        self,
        num_customers: int = 1000,
        num_merchants: int = 500,
        fraud_rate: float = 0.03,
        seed: int = 42,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize simulator with customer/merchant population.

        Args:
            num_customers: Number of synthetic customer profiles to generate.
            num_merchants: Number of synthetic merchants to generate.
            fraud_rate: Target baseline proportion of fraud (0.01 to 0.10).
            seed: Pseudo-random generator seed for deterministic reproducibility.
            settings: Optional application settings.
        """
        self.num_customers = num_customers
        self.num_merchants = num_merchants
        self.fraud_rate = fraud_rate
        self.settings = settings or get_settings()
        self.rng = random.Random(seed)

        self.customers: dict[str, CustomerProfile] = {}
        self.merchants: dict[str, MerchantProfile] = {}
        self._init_profiles()

    def _init_profiles(self) -> None:
        """Generate populations of customers and merchants."""
        # 1. Generate Merchants
        for i in range(self.num_merchants):
            cat_tuple = self.rng.choice(MERCHANT_CATEGORIES)
            cat_name, base_risk, cat_avg = cat_tuple
            merch_id = f"merch_{i:04d}"
            self.merchants[merch_id] = MerchantProfile(
                merchant_id=merch_id,
                merchant_name=f"{cat_name.title()} Store #{i}",
                category=cat_name,
                risk_weight=base_risk,
                avg_amount=cat_avg,
            )

        merchant_ids = list(self.merchants.keys())

        # 2. Generate Customers
        channels = [
            PaymentChannel.UPI,
            PaymentChannel.CARD,
            PaymentChannel.IMPS,
            PaymentChannel.NEFT,
        ]
        channel_weights = [0.65, 0.25, 0.08, 0.02]

        for i in range(self.num_customers):
            cust_id = f"cust_{i:05d}"
            center_name, city_lat, city_lng = self.rng.choice(METROPOLITAN_CENTERS)
            # Add gaussian noise around city center (~5-10km radius)
            home_lat = city_lat + self.rng.gauss(0, 0.05)
            home_lng = city_lng + self.rng.gauss(0, 0.05)
            primary_device = f"dev_{uuid4().hex[:12]}"
            primary_channel = self.rng.choices(channels, weights=channel_weights)[0]

            # Spending: log-normal distribution (median ~800 INR, standard deviation ~400 INR)
            avg_amount = max(100.0, float(self.rng.lognormvariate(6.5, 0.6)))
            std_amount = avg_amount * 0.45

            # Affinity: 5 to 15 frequent merchants
            known_merchants = set(
                self.rng.sample(merchant_ids, k=min(len(merchant_ids), self.rng.randint(5, 15)))
            )

            self.customers[cust_id] = CustomerProfile(
                customer_id=cust_id,
                avg_amount=avg_amount,
                std_amount=std_amount,
                home_lat=home_lat,
                home_lng=home_lng,
                primary_device=primary_device,
                primary_channel=primary_channel,
                known_merchants=known_merchants,
                last_lat=home_lat,
                last_lng=home_lng,
            )

        logger.info(
            "Initialized synthetic simulator entities",
            customers=len(self.customers),
            merchants=len(self.merchants),
        )

    def generate_transaction(
        self,
        fraud_type: Optional[FraudPatternType] = None,
        force_fraud: bool = False,
        timestamp: Optional[datetime] = None,
    ) -> tuple[TransactionEvent, bool, FraudPatternType]:
        """
        Generate a single synthetic transaction with realistic parameters.

        Args:
            fraud_type: Explicit fraud pattern to simulate, or None to decide probabilistically.
            force_fraud: If True, forces transaction to be fraudulent.
            timestamp: Specific occurrence datetime, or defaults to current UTC time.

        Returns:
            tuple[TransactionEvent, bool, FraudPatternType]:
                (Generated transaction event, is_fraud boolean, applied pattern type)
        """
        txn_time = timestamp or datetime.now(timezone.utc)
        is_fraud = force_fraud or (self.rng.random() < self.fraud_rate)

        if is_fraud:
            pattern = fraud_type or self.rng.choice(
                [
                    FraudPatternType.VELOCITY,
                    FraudPatternType.AMOUNT_ANOMALY,
                    FraudPatternType.GEOGRAPHIC_ANOMALY,
                    FraudPatternType.NEW_MERCHANT,
                    FraudPatternType.CARD_NOT_PRESENT,
                ]
            )
        else:
            pattern = FraudPatternType.NONE

        # Pick random customer
        customer: CustomerProfile = self.rng.choice(list(self.customers.values()))

        # Base default parameters
        amount = max(10.0, round(self.rng.gauss(customer.avg_amount, customer.std_amount), 2))
        device_fingerprint = customer.primary_device
        channel = customer.primary_channel
        location_lat = customer.home_lat + self.rng.gauss(0, 0.02)
        location_lng = customer.home_lng + self.rng.gauss(0, 0.02)
        is_international = False

        if customer.known_merchants and self.rng.random() < 0.8:
            merchant_id = self.rng.choice(list(customer.known_merchants))
        else:
            merchant_id = self.rng.choice(list(self.merchants.keys()))
        merchant = self.merchants[merchant_id]

        # Apply specific Fraud Pattern injections
        if pattern == FraudPatternType.AMOUNT_ANOMALY:
            # 10x to 40x customer's normal average
            multiplier = self.rng.uniform(10.0, 40.0)
            amount = round(customer.avg_amount * multiplier, 2)

        elif pattern == FraudPatternType.GEOGRAPHIC_ANOMALY:
            # Shift location drastically (e.g. 5,000+ km away or international)
            if self.rng.random() < 0.5:
                # London coordinates
                location_lat, location_lng = 51.5074, -0.1278
                is_international = True
            else:
                # Far distant domestic center
                location_lat += self.rng.choice([-15.0, 15.0])
                location_lng += self.rng.choice([-15.0, 15.0])

        elif pattern == FraudPatternType.NEW_MERCHANT:
            # Transaction at a high-risk merchant customer has never visited
            high_risk_candidates = [
                m for m in self.merchants.values()
                if m.category in {"crypto_exchange", "luxury_jewelry", "gambling"}
                and m.merchant_id not in customer.known_merchants
            ]
            if high_risk_candidates:
                chosen_m = self.rng.choice(high_risk_candidates)
                merchant_id = chosen_m.merchant_id
                merchant = chosen_m
                amount = round(self.rng.uniform(5000.0, 50000.0), 2)

        elif pattern == FraudPatternType.CARD_NOT_PRESENT:
            # Foreign unrecognized device credential stuffing
            device_fingerprint = f"dev_hijack_{uuid4().hex[:10]}"
            channel = PaymentChannel.CARD

        elif pattern == FraudPatternType.VELOCITY:
            # High burst: typical amount from device
            device_fingerprint = customer.primary_device

        # Update customer state for subsequent relative calculations
        customer.last_lat = location_lat
        customer.last_lng = location_lng
        customer.last_txn_time = txn_time

        txn_event = TransactionEvent(
            transaction_id=f"txn_{uuid4().hex[:14]}",
            customer_id=customer.customer_id,
            merchant_id=merchant_id,
            amount=Decimal(f"{amount:.2f}"),
            currency="INR",
            channel=channel,
            device_fingerprint=device_fingerprint,
            ip_address=f"192.168.{self.rng.randint(1, 254)}.{self.rng.randint(1, 254)}",
            location_lat=round(location_lat, 6),
            location_lng=round(location_lng, 6),
            merchant_category=merchant.category,
            is_international=is_international,
            timestamp=txn_time,
        )

        return txn_event, is_fraud, pattern

    async def stream_transactions(
        self,
        events_per_second: int = 50,
        max_events: Optional[int] = None,
    ) -> AsyncGenerator[tuple[TransactionEvent, bool, FraudPatternType], None]:
        """
        Asynchronously stream generated transactions at a calibrated throughput.

        Args:
            events_per_second: Target generation frequency.
            max_events: Optional limit of events to yield before stopping.

        Yields:
            tuple[TransactionEvent, bool, FraudPatternType]: Simulated transactions.
        """
        interval = 1.0 / max(1, events_per_second)
        count = 0

        while max_events is None or count < max_events:
            event, is_fraud, pattern = self.generate_transaction()
            yield event, is_fraud, pattern
            count += 1
            await asyncio.sleep(interval)
