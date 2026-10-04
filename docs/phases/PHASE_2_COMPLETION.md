# Phase 2 Completion: Transaction Simulator & Synthetic Data Generation

## 1. Overview & Objective
Phase 2 delivers the synthetic ingestion engine powering **SentinelForge**. It models realistic payment behaviors across credit, debit, and UPI payment rails with a ~2.5% background fraud rate and 5 explicit fraud attack patterns. It also provides the historical seeding script required for offline feature snapshot extraction and LightGBM model training.

---

## 2. Artifacts & Deliverables Completed

| Component | Path | Description & Design Justification |
| :--- | :--- | :--- |
| **Transaction Simulator** | [`services/transaction_simulator/simulator.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/transaction_simulator/simulator.py) | Generates synthetic transactions with customer spend baselines and merchant risk weights. |
| **Seeding Script** | [`scripts/seed_transactions.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/scripts/seed_transactions.py) | Generates time-ordered historical transactions (Parquet export & optional Postgres insert). |
| **Simulator Tests** | [`tests/unit/test_simulator.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/tests/unit/test_simulator.py) | 7 unit tests covering all 5 fraud typologies and async streaming. |

---

## 3. The 5 Fraud Typologies Simulated

1. **Velocity Bursts (`velocity`)**:
   - Successive transactions originating from the same device or customer within minutes.
   - Triggers downstream Redis sliding-window counters (`txn_count_1min`, `txn_count_5min`).
2. **Amount Anomaly (`amount_anomaly`)**:
   - Single transactions 10x–40x larger than the customer's historical mean spend.
   - Triggers downstream rolling z-score feature (`amount_zscore > 3.0`).
3. **Geographic Anomaly (`geographic_anomaly`)**:
   - Sudden location displacements (thousands of kilometers away) or cross-border spoofing (`is_international = True`).
   - Triggers Haversine distance computations against customer's anchor location.
4. **New High-Risk Merchant Fraud (`new_merchant`)**:
   - First-time customer visits to high-risk merchant categories (crypto exchanges, online gambling, luxury jewelry).
   - Triggers `is_new_merchant` and `merchant_fraud_rate` features.
5. **Card-Not-Present Device Mismatch (`card_not_present`)**:
   - Online payments originating from unrecognized device fingerprints and IP subnets (simulating credential stuffing).
   - Triggers `is_new_device` and `unique_devices_1hr`.

---

## 4. Key Design Decisions & Interview Talking Points

### 4.1 Non-Uniform Entity Modeling vs. Random Noise
- **Decision**: Modeled 1,000 distinct customer entities with log-normal spending distributions, home city coordinates, and known merchant affinity graphs.
- **Interview Talking Point**:
  > *"Real financial fraud models fail when trained on uniform random noise because fraud is defined as a deviation from an established behavioral baseline. By anchoring each customer to an authentic spending distribution and home GPS anchor, features like Haversine distance and amount z-score produce statistically meaningful predictive signals."*

### 4.2 Class Imbalance Strategy
- **Decision**: Hardcoded configurable fraud rates (2.5% default) matching real banking fraud ratios.
- **Interview Talking Point**:
  > *"In payments, fraud represents 1-3% of traffic. Building the simulator with realistic class imbalance forces the downstream ML pipeline to address the precision-recall trade-off directly via scale_pos_weight and stratified evaluation rather than relying on artificial 50/50 balance."*

---

## 5. Verification & Test Execution Results

```text
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\AI_ML_Bootcamp\2. SentinelForge
configfile: pyproject.toml
plugins: anyio-4.15.1, asyncio-1.4.0, mock-3.16.0
asyncio: mode=Mode.AUTO
collected 25 items

tests\integration\test_postgres.py ..                                    [  8%]
tests\integration\test_redis.py .                                        [ 12%]
tests\unit\test_models.py .........                                      [ 48%]
tests\unit\test_settings.py ......                                       [ 72%]
tests\unit\test_simulator.py .......                                     [100%]

============================= 25 passed in 0.38s ==============================
```

---

## 6. Next Steps
With Phase 2 verified and the historical transaction dataset generating, we proceed to **Phase 3: Streaming Feature Engine**, building the single source of truth feature definitions, sliding-window velocity aggregators, and the dual online/offline store synchronization.
