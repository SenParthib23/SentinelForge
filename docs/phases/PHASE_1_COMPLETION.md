# Phase 1 Completion: Foundation & Shared Infrastructure

## 1. Overview & Objective
Phase 1 establishes the rock-solid architectural foundation for **SentinelForge**, ensuring that all downstream real-time streaming services (Layers 1–6) and offline ML lifecycle components (Layers 7–10) share a unified, validated data model, typed configuration system, production-grade observability metrics, and asynchronous database managers.

---

## 2. Artifacts & Deliverables Completed

| Component | Path | Description & Design Justification |
| :--- | :--- | :--- |
| **Directory Skeleton** | `shared/`, `services/`, `ml/`, `infra/`, `tests/`, `scripts/` | Layered modular architecture isolating real-time auth path from async workflows. |
| **Docker Infrastructure** | [`docker-compose.yml`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/docker-compose.yml) | Multi-container stack (Zookeeper, Kafka, Postgres, Redis, MLflow, Prometheus, Grafana). |
| **Database Schema** | [`infra/postgres/init.sql`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/infra/postgres/init.sql) | DDL for raw transactions, feature snapshots, fraud scores, cases, models, and labeled data. |
| **Pydantic Settings** | [`shared/config/settings.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/shared/config/settings.py) | Strongly-typed environment configs with `@computed_field` connection DSNs. |
| **Async PostgreSQL** | [`shared/db/postgres.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/shared/db/postgres.py) | High-performance `asyncpg` connection pool with query helpers and health probes. |
| **Async Redis Client** | [`shared/db/redis_client.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/shared/db/redis_client.py) | `redis.asyncio` client with Hash maps and Sorted Sets for sliding-window velocity. |
| **Structured Logging** | [`shared/logging/logger.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/shared/logging/logger.py) | `structlog` with contextvars injecting `request_id` and `transaction_id`. |
| **Prometheus Metrics** | [`shared/metrics/prometheus.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/shared/metrics/prometheus.py) | Custom latency histograms (<150ms SLA), counters, and deployment gauges. |
| **Data Models** | [`shared/models/`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/shared/models/) | Canonical schemas: `TransactionEvent`, `FeatureVector`, `FraudCase`, `ModelVersion`. |
| **Unit & Integration Tests** | [`tests/`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/tests/) | 18 passing tests verifying settings, model validation, and DB clients. |

---

## 3. Key Design Decisions & Interview Talking Points

### 3.1 Port Collision Avoidance with MAOP
- **Decision**: Configured PostgreSQL on **5433** (vs 5432), Redis on **6380** (vs 6379), Prometheus on **9091** (vs 9090), Grafana on **3002** (vs 3000), and MLflow on **5001** (vs 5000).
- **Interview Value**: Shows practical operational awareness when co-locating multiple distributed systems on shared development or CI runners without resource collisions.

### 3.2 Dual Online/Offline Feature Store Schemas
- **Decision**: Defined `FeatureVector` as the immutable representation of computed features.
- **Interview Value**: Eliminates **training-serving skew**. Online services retrieve Redis hash fields into `FeatureVector`, and batch training jobs query Postgres `feature_snapshots` into `FeatureVector`, guaranteeing identical column order and types.

### 3.3 Redis Sorted Sets for Velocity Counters
- **Decision**: In `RedisClient`, implemented `zadd`, `zcount`, and `zremrangebyscore`.
- **Interview Value**: For calculating transactions over sliding time windows (e.g. 5 minutes, 1 hour), sorted sets using transaction timestamps as scores provide $O(\log N)$ range counting and atomic sliding window eviction, eliminating stateful in-memory counters in Python worker nodes.

### 3.4 Custom Non-Linear Prometheus Buckets
- **Decision**: Scored latency histogram uses specific boundaries: `[1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 150ms]`.
- **Interview Value**: Default exponential buckets are too coarse around the critical 10ms–150ms boundary. Because payment gateways enforce strict 150ms timeout SLAs, these custom buckets immediately detect p95 and p99 regressions before timeouts occur.

---

## 4. Verification & Test Execution Results

```text
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\AI_ML_Bootcamp\2. SentinelForge
configfile: pyproject.toml
plugins: anyio-4.15.1, asyncio-1.4.0, mock-3.16.0
asyncio: mode=Mode.AUTO
collected 18 items

tests\integration\test_postgres.py ..                                    [ 11%]
tests\integration\test_redis.py .                                        [ 16%]
tests\unit\test_models.py .........                                      [ 66%]
tests\unit\test_settings.py ......                                       [100%]

============================= 18 passed in 0.33s ==============================
```

---

## 5. Next Steps
With the foundation in place, we proceed to **Phase 2: Transaction Simulator & Kafka Producer**, implementing realistic synthetic transaction generation with 5 distinct fraud patterns and seeding 100,000 baseline transactions.
