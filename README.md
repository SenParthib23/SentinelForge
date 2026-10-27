<div align="center">

# 🛡️ SentinelForge

### Production-Grade Real-Time Transaction Fraud Detection & LLM Investigation Platform

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg?logo=python)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM%20v4.6-brightgreen.svg?logo=scikit-learn)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/Gateway-FastAPI%200.115-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Redis](https://img.shields.io/badge/Online%20Store-Redis%207.2-DC382D.svg?logo=redis)](https://redis.io/)
[![PostgreSQL](https://img.shields.io/badge/Offline%20Store-PostgreSQL%2016-336791.svg?logo=postgresql)](https://www.postgresql.org/)
[![Groq LLM](https://img.shields.io/badge/LLM-Groq%20Llama--3.1--8B-F05032.svg)](https://groq.com/)
[![Tests](https://img.shields.io/badge/Tests-90%20Passing%20(100%25)-success.svg)](https://github.com/SenParthib23/SentinelForge)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

*Built for sub-10ms payment authorization SLAs, streaming feature engineering, verifiable AI investigations, and continuous self-improving ML lifecycles.*

</div>

---

## 📌 Table of Contents
1. [System Overview & Mission](#-system-overview--mission)
2. [Dual-Path Architecture](#-dual-path-architecture)
3. [Latency Budget Breakdown (<150ms SLA)](#-latency-budget-breakdown-150ms-sla)
4. [Technology Stack](#-technology-stack)
5. [Key Subsystems](#-key-subsystems)
   - [Real-Time Feature Store (Redis + Postgres)](#1-real-time-feature-store-redis--postgres)
   - [Sub-10ms LightGBM Scoring Engine](#2-sub-10ms-lightgbm-scoring-engine)
   - [Async LLM Investigation Assist (Groq Llama-3-8B)](#3-async-llm-investigation-assist-groq-llama-3-8b)
   - [Evidence Grounding Citation Verifier](#4-evidence-grounding-citation-verifier)
   - [Model Lifecycle: Sticky Canary, Circuit Breaker & PSI Drift](#5-model-lifecycle-sticky-canary-circuit-breaker--psi-drift)
   - [4-bit QLoRA Fine-Tuning Pipeline](#6-4-bit-qlora-fine-tuning-pipeline)
6. [Interactive Analyst Cockpit & Gateway UI](#-interactive-analyst-cockpit--gateway-ui)
7. [API Specification](#-api-specification)
8. [Quickstart & Local Installation](#-quickstart--local-installation)
9. [Mathematical Formulations](#-mathematical-formulations)
10. [Test Suite Verification](#-test-suite-verification)

---

## 🚀 System Overview & Mission

Modern financial payment rails (UPI, Card Schemes, FedNow, Faster Payments) enforce strict synchronous latency budgets ($<150\text{ms}$) where high-volume transactions must be authorized or declined without user friction. Meanwhile, financial intelligence regulations (FinCEN, FIU-IND, FATF) mandate thorough, evidence-backed **Suspicious Activity Reports (SAR)** for every flagged transaction.

**SentinelForge bridges this architectural divide using a Dual-Path Design:**
- **Path 1 (Synchronous Auth Path):** A low-latency LightGBM scoring model computes fraud probabilities in **~4.5ms** using 17 streaming behavioral features fetched from a local Redis sorted-set feature store in **<1ms**.
- **Path 2 (Asynchronous Investigation Assist):** High-risk and critical transactions are held or approved, while an asynchronous Groq Llama-3-8B agent synthesizes a 360° context dossier (identity graph, device fingerprints, merchant risk, spending velocity) and drafts an audit-ready SAR narrative in background time, validated by a strict **Evidence Grounding Citation Verifier**.
- **The Retraining Flywheel:** Human analyst adjudications close the loop by automatically appending verified ground truth labels into PostgreSQL `labeled_data` for model retraining and QLoRA fine-tuning.

---

## 🏛️ Dual-Path Architecture

```mermaid
flowchart TD
    subgraph Client["Payment Ingestion & Channels"]
        POS["POS Terminals"]
        Web["E-Commerce Web"]
        UPI["Mobile UPI Apps"]
    end

    subgraph Gateway["FastAPI Gateway (<150ms SLA)"]
        Trace["X-Request-ID Tracing"]
        RateLimit["Redis Sliding-Window Rate Limiter"]
        Auth["JWT Security Authenticator"]
    end

    subgraph Path1["Path 1: Synchronous Scoring Critical Path (<10ms)"]
        Fetch["Feature Fetcher (<1ms Redis HGETALL)"]
        LGBM["LightGBM Engine (~4.5ms)"]
        Router["Threshold Router (Cost-Sensitive Matrix)"]
    end

    subgraph Decision["Authorization Outcome"]
        Approve["LOW (<0.15) -> Auto-Approve"]
        StepUp["MEDIUM (0.15-0.70) -> Step-Up 2FA"]
        Hold["CRITICAL (>0.70) -> Hold / Decline"]
    end

    subgraph Path2["Path 2: Asynchronous Intelligence Assist (Off Critical Path)"]
        CaseBuild["360° Case Builder (Entity Graph + Profile)"]
        Groq["Groq Llama-3-8B Agent (SAR Synthesis)"]
        Grounding["Evidence Grounding Verifier (Citations >= 0.85)"]
        SSE["Server-Sent Events (SSE Streamer)"]
    end

    subgraph Ops["Risk Operations & Lifecycle"]
        Cockpit["Analyst Cockpit UI (Real-Time Terminal)"]
        HumanReview["Human Analyst Adjudication"]
        Flywheel["Postgres labeled_data Flywheel"]
        Canary["Sticky Canary Router & Auto-Rollback"]
        Drift["PSI Feature Drift Monitor"]
    end

    Client --> Gateway
    Gateway --> Fetch
    Fetch --> LGBM
    LGBM --> Router
    Router --> Decision
    
    Decision -.->|If Flagged High/Critical| CaseBuild
    CaseBuild --> Groq
    Groq --> Grounding
    Grounding --> SSE
    SSE --> Cockpit
    Cockpit --> HumanReview
    HumanReview --> Flywheel
    Flywheel -.-> Canary
    Canary --> Drift
```

---

## ⏱️ Latency Budget Breakdown (<150ms SLA)

Financial payment networks drop or timeout transactions that fail to respond within 150ms. SentinelForge's synchronous path consumes only **~8.5ms (5.7% of budget)**:

| Step | Operation | SLA Allocation | Typical Observed | Technology / Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Network Gateway Ingress & TLS | 15.0 ms | 2.5 ms | FastAPI + Uvicorn Async Loop |
| **2** | Rate Limiting & Auth Check | 5.0 ms | 0.8 ms | Redis sliding-window ZREMRANGE/ZCARD |
| **3** | Feature Store Vector Retrieval | 5.0 ms | **0.8 ms** | Redis Hash `HGETALL` + Local Memory Fallback |
| **4** | LightGBM Inference Scoring | 10.0 ms | **4.5 ms** | C++ optimized tree traversal, batch size = 1 |
| **5** | Dynamic Threshold Routing | 2.0 ms | 0.1 ms | Cost-matrix calibrated lookup |
| **6** | Async Offload & Gateway Egress | 5.0 ms | 0.8 ms | `asyncio.create_task()` background handoff |
| **Total** | **End-to-End Payment Decision** | **150.0 ms** | **~8.7 ms** | **~94% Headroom Preserved** |

---

## 🛠️ Technology Stack

| Layer | Component | Technology | Role & Purpose |
| :--- | :--- | :--- | :--- |
| **Compute & Model** | Classical ML Engine | **LightGBM 4.6.0** | Tabular fraud classifier, dynamic `scale_pos_weight=38.18` |
| **Generative AI** | LLM Investigation Agent | **Groq API (Llama-3.1-8B-Instant)** | Async narrative generation, FinCEN/FIU-IND compliant SAR drafting |
| **Fine-Tuning** | Parameter-Efficient Tuning | **Hugging Face PEFT + BitsAndBytes** | 4-bit QLoRA adapter training ($r=16, \alpha=32$) on consumer GPU |
| **Online Storage** | Real-Time Feature Store | **Redis 7.2** (Port `6380`) | Sliding-window velocity counters, Redis Hashes & Sorted Sets |
| **Offline Storage** | Warehouse & Ground Truth | **PostgreSQL 16** (Port `5433`) | Transaction logs, case dossiers, `labeled_data` retraining store |
| **API Gateway** | REST & SSE Server | **FastAPI 0.115** | Async HTTP router, JWT auth, SSE live stream |
| **Observability** | Metrics & Monitoring | **Prometheus + Grafana** (Ports `9091`, `3002`) | Latency histograms, drift metrics, canary performance |
| **Data Engine** | Streaming Feature Engine | **NumPy, Pandas, SciPy** | 17 SSOT features, Haversine velocity, Wasserstein drift |

---

## 🔬 Key Subsystems

### 1. Real-Time Feature Store (Redis + Postgres)
- **17 Single Source of Truth (SSOT) Features:** Defined once in `feature_definitions.py` and shared identically across training and inference.
- **Sliding-Window Velocity:** Implemented via Redis Sorted Sets (`ZADD`, `ZREMRANGEBYSCORE`, `ZCOUNT`). Tracks transaction counts and monetary sums over 1-minute, 5-minute, and 1-hour windows in $O(\log N)$ time.
- **Geographic Impossible Travel:** Haversine great-circle distance divided by $\Delta t$ between consecutive transactions. Flags physical velocity $> 800\text{ km/h}$.
- **Bipartite Identity Graph:** Card-Device-IP adjacency tracking to identify coordinated mule account syndicates.

### 2. Sub-10ms LightGBM Scoring Engine
- **Class Imbalance Calibration:** Real-world fraud rate is ~2.5%. Model is trained with dynamic `scale_pos_weight=38.18` to heavily penalize False Negatives.
- **Cost-Sensitive Threshold Optimization:** Evaluates asymmetric cost surface:
  $$\text{Total Cost} = C_{\text{FN}} \cdot \text{FN} + C_{\text{FP}} \cdot \text{FP} \quad (C_{\text{FN}} = \$500, \; C_{\text{FP}} = \$10)$$
- **Zero-Downtime Hot-Reloading:** Atomic pointer swapping loads updated LightGBM `.txt` weights without dropping active socket connections.

### 3. Async LLM Investigation Assist (Groq Llama-3-8B)
- **360° Context Dossier:** Assembles cardholder history, spending variance, linked identity graph nodes, and merchant risk profiles.
- **Structured SAR Reporting:** Automatically structures reports into:
  1. *Transaction Summary*
  2. *Categorized Risk Indicators* ([STRONG], [MODERATE], [WEAK])
  3. *Customer Historical Spending Context*
  4. *Recommended Action* (`APPROVE`, `HOLD_FOR_REVIEW`, `ESCALATE_URGENT`)

### 4. Evidence Grounding Citation Verifier
- **Zero Hallucination Policy:** Every empirical claim made by the LLM must feature bracketed citations, e.g. `[Citing: key=value]`.
- **Automated Verification Gate:** Cross-checks cited attributes and numerical quantities against the ground-truth JSON context within a 5% tolerance window.
- **Fidelity Score Gate:** If Grounding Fidelity drops below $0.85$, the narrative is discarded and an automated deterministic fallback is generated.

### 5. Model Lifecycle: Sticky Canary, Circuit Breaker & PSI Drift
- **Sticky Canary Hashing:** Uses `MD5(customer_id) % 100` to partition canary traffic. Guarantees that rapid successive transactions from the same cardholder hit the identical model version.
- **Automated Circuit Breaker Rollback:** Continuously tracks canary health in a sliding window. Instantly reverts canary split to 0% if:
  - P99 Latency $> 10.0\text{ ms}$
  - Error Rate $> 0.5\%$
  - Anomaly Flag Rate $> 3\times$ Champion baseline
- **Covariate Shift Drift Monitor:** Computes Population Stability Index (PSI) and Wasserstein Distance across all 17 features:
  $$\text{PSI} = \sum \left( P_i - Q_i \right) \times \ln\left(\frac{P_i}{Q_i}\right)$$

### 6. 4-bit QLoRA Fine-Tuning Pipeline
- **Parameter-Efficient Tuning:** Quantizes Llama-3-8B base model into 4-bit NormalFloat (NF4) with Double Quantization.
- **LoRA Hyperparameters:** Injects low-rank adapters ($r=16, \alpha=32$) across all linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Resource Footprint:** 16.78M trainable parameters (0.21% of base). Runs comfortably on single consumer GPUs with ~5.4GB VRAM.

---

## 🖥️ Interactive Analyst Cockpit & Gateway UI

SentinelForge includes a real-time risk operations cockpit served directly from the FastAPI gateway:
- **Cockpit URL:** `http://localhost:8000/dashboard` (or `http://localhost:8000/`)
- **Interactive OpenAPI Documentation:** `http://localhost:8000/docs`
- **Prometheus Metrics Exporter:** `http://localhost:8000/metrics`

### Dashboard Features:
- **Live SSE Event Feed:** Auto-updates as transactions are scored and flagged.
- **Grounded Evidence Viewer:** Inspects LLM SAR narratives with interactive citation highlights.
- **One-Click Adjudication:** Analysts can Approve or Reject cases, directly closing the Postgres retraining loop.
- **Canary & Drift Telemetry:** Live traffic split adjustment and PSI feature drift dials.

---

## 📡 API Specification

| Method | Endpoint | Description | SLA / Latency |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/auth/token` | Issues OAuth2 JWT bearer token | < 5 ms |
| `POST` | `/api/v1/score` | **Synchronous transaction fraud scoring** | **< 10 ms** |
| `GET` | `/api/v1/cases` | Retrieves pending fraud cases for analyst review | < 25 ms |
| `GET` | `/api/v1/cases/{case_id}` | Retrieves full 360° case dossier and LLM narrative | < 15 ms |
| `POST` | `/api/v1/cases/{case_id}/feedback` | **Submits analyst adjudication and updates `labeled_data`** | < 20 ms |
| `GET` | `/api/v1/cases/stream` | Server-Sent Events (SSE) live push stream | Real-time |
| `POST` | `/api/v1/models/canary/split` | Updates canary traffic split percentage (0-100%) | < 5 ms |
| `GET` | `/api/v1/models/canary/status` | Returns active canary configuration & health metrics | < 2 ms |
| `POST` | `/api/v1/models/{name}/rollback` | Triggers immediate manual or circuit-breaker rollback | < 1 ms |
| `GET` | `/api/v1/models/drift/status` | Returns latest 17-feature PSI and Wasserstein drift report | < 30 ms |
| `GET` | `/health` | Liveness and dependency readiness probe | < 2 ms |

---

## ⚡ Quickstart & Local Installation

### Prerequisites
- Python 3.12 or 3.13
- Docker & Docker Compose (optional for local Redis/Postgres)
- Groq API Key (optional; deterministic fallback active if omitted)

### 1. Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/SenParthib23/SentinelForge.git
cd SentinelForge

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your GROQ_API_KEY if desired.
```

### 3. Start Containerized Infrastructure (Isolated Ports)
```bash
docker compose up -d
```
*Port Allocations:*
- PostgreSQL: `localhost:5433`
- Redis: `localhost:6380`
- Prometheus: `localhost:9091`
- Grafana: `localhost:3002`

### 4. Run the Live End-to-End System Demonstration
Experience the complete dual-path platform in a single terminal run:
```bash
python scripts/run_demo.py
# Or run with immediate fast transitions:
python scripts/run_demo.py --fast
```

### 5. Launch the FastAPI Gateway & Analyst Cockpit
```bash
python -m uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **[http://localhost:8000/dashboard](http://localhost:8000/dashboard)** in your browser!

---

## 📐 Mathematical Formulations

### Cost-Sensitive Optimization
$$\mathcal{L}(T) = \sum_{i=1}^N \left[ C_{\text{FN}} \cdot y_i \cdot \mathbb{I}(\hat{y}_i(T) = 0) + C_{\text{FP}} \cdot (1 - y_i) \cdot \mathbb{I}(\hat{y}_i(T) = 1) \right]$$

### Population Stability Index (PSI)
$$\text{PSI} = \sum_{k=1}^K \left( \text{Actual}_k - \text{Expected}_k \right) \times \ln\left(\frac{\text{Actual}_k}{\text{Expected}_k}\right)$$
- $\text{PSI} < 0.10$: Stable; no action required.
- $0.10 \le \text{PSI} < 0.25$: Moderate drift; warning triggered.
- $\text{PSI} \ge 0.25$: Critical drift; automatic retraining event emitted.

### Haversine Great-Circle Velocity
$$d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right), \quad v = \frac{d}{\Delta t}$$

---

## 🧪 Test Suite Verification

SentinelForge maintains a **100% test pass rate** with 90 automated unit and integration tests:

```bash
python -m pytest tests/
```

```text
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
collected 90 items

tests/integration/test_feature_pipeline.py ..                            [  2%]
tests/integration/test_fine_tuning_pipeline.py .                         [  3%]
tests/integration/test_investigation_pipeline.py .                       [  4%]
tests/integration/test_model_lifecycle.py .                              [  5%]
tests/integration/test_postgres.py ..                                    [  7%]
tests/integration/test_redis.py .                                        [  8%]
tests/integration/test_scoring_pipeline.py .                             [ 10%]
tests/integration/test_training_pipeline.py .                            [ 11%]
tests/unit/test_api_auth.py ...                                          [ 14%]
tests/unit/test_api_health_and_models.py ...                             [ 17%]
tests/unit/test_api_investigation.py ...                                 [ 21%]
tests/unit/test_api_scoring.py .                                         [ 22%]
tests/unit/test_auto_rollback.py ...                                     [ 25%]
tests/unit/test_canary_router.py .....                                   [ 31%]
tests/unit/test_case_builder.py ..                                       [ 33%]
tests/unit/test_class_imbalance.py ...                                   [ 36%]
tests/unit/test_data_loader.py ..                                        [ 38%]
tests/unit/test_dataset_curator.py ...                                   [ 42%]
tests/unit/test_drift_detector.py ...                                    [ 45%]
tests/unit/test_eval_metrics.py ...                                      [ 48%]
tests/unit/test_evidence_grounding.py ...                                [ 52%]
tests/unit/test_feature_definitions.py .......                           [ 60%]
tests/unit/test_llm_eval.py ...                                          [ 63%]
tests/unit/test_models.py .........                                      [ 73%]
tests/unit/test_qlora_config.py ..                                       [ 75%]
tests/unit/test_regression_eval.py ..                                    [ 77%]
tests/unit/test_scorer.py ...                                            [ 81%]
tests/unit/test_settings.py ......                                       [ 87%]
tests/unit/test_simulator.py .......                                     [ 95%]
tests/unit/test_threshold_router.py ....                                 [100%]

============================= 90 passed in 8.35s ==============================
```

---

<div align="center">
  <sub>Engineered by Parthib Sen • Real-Time Fraud & Machine Learning Systems</sub>
</div>
