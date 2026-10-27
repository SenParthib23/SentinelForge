# SentinelForge — Technical Interview Guide & Architecture Deep-Dive

> **Target Roles:** Senior / Staff Machine Learning Engineer, MLOps Architect, Lead AI Systems Engineer, Financial Risk Systems Designer.  
> **Repository:** [SentinelForge (GitHub)](https://github.com/SenParthib23/SentinelForge)

---

## 📌 Executive Pitch (The 60-Second Elevator Pitch)

> *"SentinelForge is an end-to-end real-time fraud prevention and AI investigation platform designed for modern high-velocity payment rails (like UPI or Credit Card schemes) with a strict `<150ms` authorization SLA.*
> 
> *Because LLMs are fundamentally too slow (`800ms–2000ms`) for the authorization critical path, SentinelForge uses a **Dual-Path Architecture**:*
> 1. *A synchronous path where LightGBM scores transactions in **~4.5ms** using 17 streaming behavioral features fetched from Redis in **<1ms**.*
> 2. *An asynchronous path where flagged transactions trigger Groq Llama-3-8B in the background to synthesize a 360° context dossier (identity graph, device fingerprints, geographic impossible travel) and draft a regulatory-compliant Suspicious Activity Report (SAR).*
> 
> *To eliminate LLM hallucinations in financial audits, our **Evidence Grounding Engine** parses bracketed citation tokens from the narrative and mathematically cross-checks them against raw JSON context within a 5% tolerance window. If fidelity falls below 85%, an automated fallback takes over. Human analyst adjudications close the loop by automatically appending verified ground-truth labels into PostgreSQL `labeled_data` for continuous retraining, monitored by sticky canary traffic splitting, automated SLA circuit breakers, and PSI feature drift monitors."*

---

## 🧠 Core System Design & MLOps Questions

### Q1: Why not run an LLM directly during payment authorization?
**Interview Context:** Tests understanding of real-world payment SLAs, tail latencies, and compute economics.

* **Answer:**
  * Payment gateways enforce a hard `<150ms` end-to-end timeout. If a response is not received, the transaction times out, leading to cart abandonment and merchant financial penalties.
  * State-of-the-art quantized LLMs running on specialized inference accelerators (e.g. Groq LPU or vLLM on H100) have a time-to-first-token (TTFT) of ~100ms and generation latencies between **800ms and 2500ms** for a 300-word narrative.
  * In contrast, LightGBM evaluates decision tree splits via compiled C++ routines, scoring a 17-feature vector in **~4.5ms** on ordinary CPU cores.
  * SentinelForge places **LightGBM on the synchronous auth critical path** and offloads **Llama-3-8B to the asynchronous investigation path** via non-blocking background workers (`asyncio.create_task()` or Kafka event streaming).

---

### Q2: How do you prevent train/serve feature skew in high-velocity streaming?
**Interview Context:** Evaluates feature store design, real-time streaming architectures, and temporal leakage prevention.

* **Answer:**
  * **Single Source of Truth (SSOT):** In `shared/models/feature_definitions.py`, we define all 17 feature schemas, mathematical boundaries, and compute functions once.
  * **Dual-Store Architecture:**
    * **Online Store (Redis):** Features are written immediately upon event receipt using Redis Hashes (`features:{txn_id}`) with a 24-hour TTL, enabling `<1ms` point lookups during payment scoring.
    * **Offline Store (PostgreSQL / Parquet):** Transaction records and point-in-time feature snapshots are appended for training batch jobs.
  * **Sliding Window Velocity via Redis Sorted Sets:**
    * Rather than querying a relational database, customer transaction timestamps are stored as scores in Redis Sorted Sets (`velocity:{customer_id}:txns`).
    * Counting transactions within a 5-minute window is a single `ZCOUNT` query in $O(\log N + M)$ time (~0.4ms).
    * Atomic cleanup is performed using `ZREMRANGEBYSCORE`, eliminating memory leaks.

---

### Q3: Fraud datasets suffer from extreme class imbalance (~2.5%). How does SentinelForge handle this without naive oversampling?
**Interview Context:** Tests applied statistics, cost-sensitive learning, and business metric alignment.

* **Answer:**
  * **Dynamic `scale_pos_weight`:**
    $$\text{scale\_pos\_weight} = \frac{N_{\text{neg}}}{N_{\text{pos}}} = \frac{97,450}{2,550} \approx 38.18$$
    This scales the gradient of the minority (fraud) class in LightGBM's objective loss function, preventing the model from defaulting to predicting the majority class.
  * **Asymmetric Cost-Sensitive Optimization:**
    In payment fraud, a **False Negative (missed fraud)** costs an average of **$500** (chargeback fee + merchandise loss), whereas a **False Positive (unnecessary 2FA prompt)** costs only **$10** (customer friction).
    $$\mathcal{L}(T) = C_{\text{FN}} \cdot \text{FN}(T) + C_{\text{FP}} \cdot \text{FP}(T)$$
    Instead of using the default 0.5 threshold, SentinelForge evaluates 1,000 threshold candidates across validation slices to select the optimal operational cutoff that minimizes expected financial dollar loss.
  * **Precision-Recall AUC (PR-AUC):** We evaluate PR-AUC and Recall@1% FPR rather than ROC-AUC, because ROC-AUC is misleadingly optimistic under severe class imbalance.

---

### Q4: How do you guarantee the LLM will not hallucinate facts in a regulatory SAR report?
**Interview Context:** Critical for enterprise AI compliance, financial audits, and FinCEN / FIU-IND regulations.

* **Answer:**
  * **Evidence Grounding Citation Syntax:** The LLM prompt explicitly enforces bracketed citations for every factual assertion:  
    `[Citing: key=value]` (e.g. `[Citing: distance_from_last_txn=850.5, is_international=true]`).
  * **Automated Evidence Verifier (`EvidenceGroundingVerifier`):**
    1. Extracts all citation tokens via regex parsing.
    2. Flattens the raw JSON context bundle.
    3. Verifies that the cited field exists in the context bundle.
    4. For numerical values, enforces mathematical proximity check within a 5% tolerance window:
       $$\left|\frac{V_{\text{claim}} - V_{\text{ground\_truth}}}{V_{\text{ground\_truth}}}\right| \le 0.05$$
    5. Computes a **Grounding Fidelity Score**:
       $$\text{Fidelity} = \frac{\text{Valid Citations}}{\text{Total Citations}}$$
  * **Safety Fallback:** If Fidelity drops below **0.85**, the model output is rejected, an alert is logged, and a deterministic template-based SAR is generated to guarantee auditability.

---

### Q5: How do you deploy and validate a new model candidate in production safely?
**Interview Context:** Tests production deployment strategies, canary routing, and automated rollback systems.

* **Answer:**
  * **Two-Gate Promotion:**
    * *Gate 1 (Static Benchmark):* Candidate must surpass absolute operational baselines (ROC-AUC $\ge 0.85$, PR-AUC $\ge 0.65$, P99 Latency $< 10\text{ms}$).
    * *Gate 2 (Head-to-Head Showdown):* Candidate and Production Champion are evaluated on the exact same held-out test slice. The candidate must not regress by $>1.5\%$ on Recall or $>2.0\%$ on PR-AUC.
  * **Sticky Customer Canary Routing:**
    * Random percentage routing causes customer friction (a cardholder making 2 rapid transactions might hit two different models with conflicting thresholds).
    * We hash `customer_id` using MD5 modulo 100:
      $$H = \text{MD5}(\text{customer\_id}) \pmod{100} < \text{canary\_percent}$$
    * Guarantees deterministic stickiness for each customer.
  * **Automated Circuit Breaker Rollback (`AutoRollbackManager`):**
    * Continuously monitors a sliding window of canary requests.
    * Automatically terminates canary traffic (snapping back to 100% Champion in `<1ms`) if:
      1. P99 Latency $> 10.0\text{ms}$
      2. Execution Error Rate $> 0.5\%$
      3. False Positive Rate $> 3\times$ Champion baseline

---

### Q6: How do you detect Covariate Shift (Feature Drift) before the model degrades?
**Interview Context:** Evaluates statistical monitoring, data drift metrics, and continuous retraining triggers.

* **Answer:**
  * **Population Stability Index (PSI):**
    $$\text{PSI} = \sum_{k=1}^K \left( \text{Actual}_k - \text{Expected}_k \right) \times \ln\left(\frac{\text{Actual}_k}{\text{Expected}_k}\right)$$
    * $\text{PSI} < 0.10$: Distribution is stable (Nominal).
    * $0.10 \le \text{PSI} < 0.25$: Moderate drift (Warning alert).
    * $\text{PSI} \ge 0.25$: Significant covariate shift (Triggers retraining event).
  * **Wasserstein Distance (Earth Mover's Distance):**
    Computed alongside PSI for continuous numeric features (e.g., `amount_zscore`) to measure the geometric cost of transforming the reference distribution into the current inference distribution.

---

### Q7: Why use 4-bit QLoRA instead of standard Full Fine-Tuning for the LLM?
**Interview Context:** Tests LLM parameter-efficient fine-tuning (PEFT), quantization techniques, and GPU memory math.

* **Answer:**
  * **Memory Footprint:** Full fine-tuning of an 8B parameter model in FP16 requires $>64\text{GB}$ VRAM for weights, optimizer states (AdamW requires 8 bytes per parameter), and activations, necessitating expensive multi-GPU clusters.
  * **QLoRA Quantization (NF4 + Double Quant):**
    * Base weights are frozen in 4-bit NormalFloat (NF4).
    * Low-rank adapter matrices ($r=16, \alpha=32$) are injected into all 7 linear projections (`q, k, v, o, gate, up, down`).
    * **Trainable Parameters:** Only **16.78M parameters** (0.21% of the 8.03B model) are updated.
    * **VRAM Consumption:** Total training VRAM drops to **~5.4GB**, allowing the fine-tuning pipeline to execute locally on standard consumer GPUs (e.g. RTX 3060/4060/4080/4090) or single T4 instances.

---

## 📊 Summary Architecture Matrix

| Requirement | Conventional Solution | SentinelForge Solution | Architectural Advantage |
| :--- | :--- | :--- | :--- |
| **Auth Latency SLA** | Monolithic microservice | **Dual-Path Architecture** | Sync LightGBM (~4.5ms) + Async Groq LLM |
| **Class Imbalance** | Naive SMOTE oversampling | **Cost-Sensitive Optimization** | Optimizes expected dollar loss ($C_{\text{FN}}=\$500$) |
| **Online Features** | Relational SQL aggregation | **Redis Sorted Sets (ZCOUNT)** | $O(\log N)$ sliding-window velocity in < 1ms |
| **LLM Reliability** | Prompt engineering | **Evidence Grounding Verifier** | Mathematical token verification with 5% tolerance |
| **Canary Deployment** | Random percent splitting | **MD5 Sticky Customer Hashing** | Cardholder experience consistency |
| **Safety Net** | Manual engineer rollback | **Circuit Breaker Auto-Rollback** | Sub-millisecond automated reversion on SLA breach |
| **Monitoring** | Accuracy metrics | **PSI + Wasserstein Distance** | Early warning of data distribution shifts |

---

<div align="center">
  <sub>Documented for Engineering Excellence • SentinelForge Technical Architecture</sub>
</div>
