"""
SentinelForge — shared/models/transaction.py

PURPOSE:
    Defines Pydantic models for raw transaction events emitted by payment
    channels and stored transaction records within the PostgreSQL audit log.

ARCHITECTURE POSITION:
    Layer 0 (Foundation) → Ingested by Transaction Simulator (Layer 1), consumed
    by Feature Engine (Layer 2), scored by Real-Time Scorer (Layer 3).

DATA INTEGRITY:
    Validates amounts, currencies, channels (card, upi, neft, imps), coordinates,
    and timestamps to ensure downstream feature aggregations never encounter null pointer
    or malformed numeric anomalies.
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class PaymentChannel(str, Enum):
    """Supported payment channels for fraud detection."""

    CARD = "card"
    UPI = "upi"
    NEFT = "neft"
    IMPS = "imps"


class TransactionEvent(BaseModel):
    """
    Real-time streaming event payload received over Kafka or API.

    Attributes:
        transaction_id: Unique payment gateway transaction reference.
        customer_id: Unique account or customer identifier.
        merchant_id: Merchant identifier where charge originates.
        amount: Transaction monetary amount.
        currency: ISO 4217 3-letter currency code (default: INR).
        channel: Payment rail (card, upi, neft, imps).
        device_fingerprint: Unique device or browser hash.
        ip_address: IPv4 or IPv6 client origin address.
        location_lat: Latitude coordinate of transaction.
        location_lng: Longitude coordinate of transaction.
        merchant_category: MCC or industry category (e.g., 'electronics', 'grocery').
        is_international: Whether transaction is cross-border.
        timestamp: Epoch timestamp or ISO-8601 datetime of occurrence.
    """

    transaction_id: str = Field(
        default_factory=lambda: f"txn_{uuid4().hex[:12]}",
        description="Unique transaction identifier",
    )
    customer_id: str = Field(..., description="Customer unique ID")
    merchant_id: str = Field(..., description="Merchant unique ID")
    amount: Decimal = Field(..., gt=Decimal("0.00"), description="Transaction amount")
    currency: str = Field(default="INR", max_length=3, description="Currency code")
    channel: PaymentChannel = Field(
        default=PaymentChannel.UPI, description="Payment method/channel"
    )
    device_fingerprint: str = Field(..., description="Device hardware/browser hash")
    ip_address: Optional[str] = Field(default=None, description="Client IP address")
    location_lat: Optional[float] = Field(
        default=None, ge=-90.0, le=90.0, description="Latitude"
    )
    location_lng: Optional[float] = Field(
        default=None, ge=-180.0, le=180.0, description="Longitude"
    )
    merchant_category: str = Field(
        default="general_retail", description="Merchant category code or description"
    )
    is_international: bool = Field(
        default=False, description="Whether transaction crosses international borders"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Transaction occurrence timestamp",
    )

    @field_validator("currency", mode="before")
    @classmethod
    def validate_currency_uppercase(cls, v: Any) -> str:
        """Ensure currency code is trimmed and standardized uppercase."""
        if isinstance(v, str):
            return v.strip().upper()
        return v


class TransactionRecord(TransactionEvent):
    """
    Persisted transaction record representation stored in PostgreSQL.

    Attributes:
        id: Primary database UUID.
        created_at: Database insertion timestamp.
    """

    id: UUID = Field(default_factory=uuid4, description="Primary database record UUID")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Audit table creation timestamp",
    )
