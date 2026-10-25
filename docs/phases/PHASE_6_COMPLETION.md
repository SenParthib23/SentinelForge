# Phase 6 Completion: FastAPI Gateway & Analyst Dashboard (SSE Streaming)

## 1. Overview & Objective
Phase 6 delivers the production API gateway of **SentinelForge**. Built with **FastAPI**, it serves as the unified interface connecting payment gateway clients to the synchronous sub-10ms scoring engine, and fraud analysts to the asynchronous case investigation dossier. It provides JWT Bearer authentication, Redis token-bucket rate limiting, structured access logging with correlation IDs, Prometheus metric exposition, and Server-Sent Events (SSE) for live investigation streaming.

---

## 2. Artifacts & Deliverables Completed

| Component | Path | Description & Design Justification |
| :--- | :--- | :--- |
| **API Entrypoint & Lifespan** | [`services/api/main.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/api/main.py) | Application factory with async lifespan management (PostgreSQL pool, Redis, LightGBM model preloading), CORS, and OpenAPI docs at `/docs`. |
| **Dependencies & Security** | [`services/api/dependencies.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/api/dependencies.py) | JWT authentication (`HS256`), access token verification, and dependency injection singletons. |
| **Access Logger Middleware** | [`services/api/middleware/request_logger.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/api/middleware/request_logger.py) | Injects unique `X-Request-ID` correlation headers, propagates contextvars, and records HTTP latency. |
| **Rate Limiter Middleware** | [`services/api/middleware/rate_limiter.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/api/middleware/rate_limiter.py) | Redis sliding-window rate limiter enforcing 60 req/min per client IP with HTTP 429 backoff. |
| **Auth Router** | [`services/api/routers/auth_router.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/api/routers/auth_router.py) | `POST /api/v1/auth/login` issuing signed JWT Bearer tokens for analysts and payment gateways. |
| **Health & Observability** | [`services/api/routers/health_router.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/api/routers/health_router.py) | `GET /api/v1/health` readiness probe and `GET /api/v1/health/metrics` for Prometheus scraping. |
| **Real-Time Scoring Router** | [`services/api/routers/scoring_router.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/api/routers/scoring_router.py) | `POST /api/v1/score` synchronous sub-10ms scoring endpoint triggering background investigation. |
| **Investigation Router** | [`services/api/routers/investigation_router.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/api/routers/investigation_router.py) | Case listing, case detail, analyst feedback (`POST .../feedback`), and SSE stream (`GET .../stream`). |
| **Model Governance Router** | [`services/api/routers/model_router.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/api/routers/model_router.py) | Model listing, champion inspection, eval-gated promotion, rollback, and drift status reports. |
| **SSE Streamer** | [`services/dashboard/sse_streamer.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/dashboard/sse_streamer.py) | In-memory Pub/Sub fan-out streaming live case milestones to connected analyst terminals. |
| **Interactive Analyst Cockpit UI** | [`services/dashboard/static/index.html`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/dashboard/static/index.html) | Glassmorphic dark dashboard with attack simulator, 17 behavioral features, live SSE alerts, and retraining feedback loop. |
| **API Test Suites** | `tests/unit/test_api_auth.py`, `test_api_scoring.py`, `test_api_investigation.py`, `test_api_health_and_models.py` | 10 tests verifying JWT auth, scoring endpoints, feedback capture, health probes, dashboard serving, and SSE dispatch. |

---

## 3. Endpoints Implemented & Verified

```http
# Web Cockpit & UI
GET    /                                   # Real-time interactive analyst cockpit dashboard
GET    /dashboard                          # Alias to analyst cockpit

# Authentication
POST   /api/v1/auth/login                  # Authenticate and receive JWT Bearer token

# Real-Time Authorization Path
POST   /api/v1/score                       # Synchronous sub-10ms LightGBM fraud scoring

# Fraud Case Management & Human Feedback Loop
GET    /api/v1/cases                       # List flagged cases (supports ?status= and ?limit=)
GET    /api/v1/cases/stream/live           # Global SSE stream for real-time alerts across all transactions
GET    /api/v1/cases/{case_id}             # Get case details with 360-degree context & narrative
POST   /api/v1/cases/{case_id}/feedback    # Submit analyst decision (approves/rejects/escalates)
GET    /api/v1/cases/{case_id}/stream      # Server-Sent Events (SSE) live progress stream for specific case

# Model Lifecycle & Governance
GET    /api/v1/models                      # List all registered models
GET    /api/v1/models/{name}/current       # Get current production champion model
POST   /api/v1/models/{name}/promote       # Trigger eval-gated candidate promotion
POST   /api/v1/models/{name}/rollback      # Execute emergency rollback to previous baseline
GET    /api/v1/models/drift/status         # Evidently AI continuous drift report

# Health & Observability
GET    /api/v1/health                      # Deep readiness probe (Postgres, Redis, Model)
GET    /api/v1/health/metrics              # Prometheus plaintext metrics scrape endpoint
```

---

## 4. Key Architectural Decisions & Interview Talking Points

### 4.1 Server-Sent Events (SSE) vs. WebSockets
- **Question**: *"Why did you use Server-Sent Events instead of WebSockets for the analyst dashboard?"*
- **Talking Point**:
  > *"Investigation assist is fundamentally unidirectional: the backend pushes asynchronous case milestones (context assembled, LLM narrative drafted, grounding verified) to the analyst's screen. SSE operates natively over standard HTTP/2, requires no special proxy or firewall configuration in corporate enterprise networks, and handles automatic client reconnection natively."*

### 4.2 Closing the Retraining Flywheel
- **Question**: *"How does the API layer support continuous model improvement?"*
- **Talking Point**:
  > *"When an analyst submits an adjudication via `POST /api/v1/cases/{id}/feedback`, the endpoint atomically updates the case status and inserts a record into the `labeled_data` table containing the original feature vector, risk score, and human ground-truth label. This continuously curates high-value labeled samples of borderline transactions for our downstream LightGBM retraining and QLoRA fine-tuning pipelines."*

---

## 5. Test Verification Results (58 Tests Passing)

```text
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
collected 58 items

tests/integration/test_feature_pipeline.py ..                            [  3%]
tests/integration/test_investigation_pipeline.py .                       [  5%]
tests/integration/test_postgres.py ..                                    [  8%]
tests/integration/test_redis.py .                                        [ 10%]
tests/integration/test_scoring_pipeline.py .                             [ 12%]
tests/unit/test_api_auth.py ...                                          [ 17%]
tests/unit/test_api_health_and_models.py ...                             [ 22%]
tests/unit/test_api_investigation.py ...                                 [ 27%]
tests/unit/test_api_scoring.py .                                         [ 29%]
tests/unit/test_case_builder.py ..                                       [ 32%]
tests/unit/test_evidence_grounding.py ...                                [ 37%]
tests/unit/test_feature_definitions.py .......                           [ 50%]
tests/unit/test_models.py .........                                      [ 65%]
tests/unit/test_scorer.py ...                                            [ 70%]
tests/unit/test_settings.py ......                                       [ 81%]
tests/unit/test_simulator.py .......                                     [ 93%]
tests/unit/test_threshold_router.py ....                                 [100%]

============================= 58 passed in 1.87s ==============================
```

---

## 6. Next Steps
With Phase 6 verified and the API layer fully operational, we are ready for **Phase 7: Offline Model Training & Regression Eval Harness**, building the stratified data loader, hyperparameter search, and the critical **Head-to-Head Regression Suite** against the current production champion.
