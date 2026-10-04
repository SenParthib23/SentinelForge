"""
SentinelForge — scripts/seed_transactions.py

PURPOSE:
    Seeds historical transaction dataset (default 100,000 transactions) with
    realistic fraud class imbalance (~2.5% fraud rate) across 5 fraud typologies.
    Persists transactions to PostgreSQL `transactions` table when database is online,
    and exports a local Apache Parquet dataset for offline ML pipeline training.

ARCHITECTURE POSITION:
    Scripts / Data Ingestion → Prepares the ground-truth baseline dataset for
    Phase 4 LightGBM training and Phase 7 ML evaluation harnesses.

INTERVIEW TALKING POINT:
    "We seeded 100k transactions exhibiting realistic 2.5% class imbalance.
    Rather than treating tabular data as independent static rows, transactions
    were generated sequentially over a 30-day temporal window with customer
    behavior histories. This ensures time-dependent features like velocity
    and rolling spend averages are non-zero and representative of live traffic."
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to sys.path for direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from services.transaction_simulator.simulator import FraudPatternType, TransactionSimulator
from shared.config.settings import get_settings
from shared.db.postgres import PostgresClient
from shared.logging.logger import configure_logging, get_logger

configure_logging(environment="development", log_level="INFO")
logger = get_logger("seed_transactions")


async def seed_historical_dataset(
    count: int = 100000,
    fraud_rate: float = 0.025,
    days: int = 30,
    export_parquet: str = "data/historical_transactions_100k.parquet",
    insert_to_db: bool = False,
) -> None:
    """
    Generate and persist temporal sequence of synthetic transactions.

    Args:
        count: Total number of transactions to generate.
        fraud_rate: Proportion of fraud transactions (e.g. 0.025 = 2.5%).
        days: Historical time span across which transactions occurred.
        export_parquet: Destination file path for Parquet output.
        insert_to_db: Whether to write records to PostgreSQL transactions table.
    """
    settings = get_settings()
    simulator = TransactionSimulator(
        num_customers=1000,
        num_merchants=500,
        fraud_rate=fraud_rate,
        seed=1337,
        settings=settings,
    )

    logger.info(
        "Starting historical transaction generation",
        total_transactions=count,
        target_fraud_rate=fraud_rate,
        time_horizon_days=days,
    )

    start_date = datetime.now(timezone.utc) - timedelta(days=days)
    time_step_seconds = (days * 86400) / max(1, count)

    records: list[dict] = []
    fraud_count = 0
    pattern_breakdown: dict[str, int] = {p.value: 0 for p in FraudPatternType}

    for i in range(count):
        txn_timestamp = start_date + timedelta(seconds=i * time_step_seconds)
        event, is_fraud, pattern = simulator.generate_transaction(timestamp=txn_timestamp)

        if is_fraud:
            fraud_count += 1
        pattern_breakdown[pattern.value] += 1

        records.append(
            {
                "transaction_id": event.transaction_id,
                "customer_id": event.customer_id,
                "merchant_id": event.merchant_id,
                "amount": float(event.amount),
                "currency": event.currency,
                "channel": event.channel.value,
                "device_fingerprint": event.device_fingerprint,
                "ip_address": event.ip_address,
                "location_lat": event.location_lat,
                "location_lng": event.location_lng,
                "merchant_category": event.merchant_category,
                "is_international": event.is_international,
                "timestamp": event.timestamp,
                "is_fraud": int(is_fraud),
                "fraud_pattern": pattern.value,
            }
        )

    df = pd.DataFrame(records)
    actual_fraud_pct = (fraud_count / count) * 100

    logger.info(
        "Completed transaction generation",
        total_generated=len(df),
        fraud_count=fraud_count,
        actual_fraud_percentage=f"{actual_fraud_pct:.2f}%",
        pattern_breakdown=pattern_breakdown,
    )

    # Save to Parquet
    parquet_path = Path(export_parquet)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info("Saved transaction dataset to Parquet", path=str(parquet_path.resolve()))

    # Insert into PostgreSQL if requested
    if insert_to_db:
        postgres = PostgresClient(settings)
        try:
            await postgres.connect()
            logger.info("Inserting transactions into PostgreSQL...")
            insert_query = """
            INSERT INTO transactions (
                transaction_id, customer_id, merchant_id, amount, currency,
                channel, device_fingerprint, ip_address, location_lat,
                location_lng, merchant_category, is_international, timestamp
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (transaction_id) DO NOTHING;
            """
            batch_size = 2000
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                args_list = [
                    (
                        r["transaction_id"],
                        r["customer_id"],
                        r["merchant_id"],
                        r["amount"],
                        r["currency"],
                        r["channel"],
                        r["device_fingerprint"],
                        r["ip_address"],
                        r["location_lat"],
                        r["location_lng"],
                        r["merchant_category"],
                        r["is_international"],
                        r["timestamp"],
                    )
                    for r in batch
                ]
                async with postgres.pool.acquire() as conn:
                    await conn.executemany(insert_query, args_list)
            logger.info("Successfully inserted transactions into PostgreSQL")
        except Exception as exc:
            logger.warning("PostgreSQL insertion skipped or failed", error=str(exc))
        finally:
            await postgres.disconnect()


def main() -> None:
    """Parse CLI arguments and run historical seeding."""
    parser = argparse.ArgumentParser(description="Seed synthetic transactions for SentinelForge.")
    parser.add_argument("--count", type=int, default=100000, help="Number of transactions to generate")
    parser.add_argument("--fraud-rate", type=float, default=0.025, help="Target fraud rate (e.g. 0.025)")
    parser.add_argument("--days", type=int, default=30, help="Temporal window in days")
    parser.add_argument(
        "--output",
        type=str,
        default="data/historical_transactions_100k.parquet",
        help="Output Parquet path",
    )
    parser.add_argument(
        "--db-insert",
        action="store_true",
        help="Attempt insertion into PostgreSQL database",
    )
    args = parser.parse_args()

    asyncio.run(
        seed_historical_dataset(
            count=args.count,
            fraud_rate=args.fraud_rate,
            days=args.days,
            export_parquet=args.output,
            insert_to_db=args.db_insert,
        )
    )


if __name__ == "__main__":
    main()
