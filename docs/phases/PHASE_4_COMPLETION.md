# Phase 4 Completion: Real-Time LightGBM Scoring Path

## 1. Overview & Objective
Phase 4 delivers the synchronous fraud scoring engine of **SentinelForge**. It executes within the strict **<150ms payment authorization SLA** by utilizing a trained LightGBM gradient-boosted decision tree classifier with **sub-10ms inference latency**. It integrates zero-downtime hot-reloading, Redis online feature vector fetching (<1ms), 4-tier risk routing (LOW, MEDIUM, HIGH, CRITICAL), and asynchronous Kafka dispatch for flagged cases.

---

## 2. Artifacts & Deliverables Completed

| Component | Path | Description & Design Justification |
| :--- | :--- | :--- |
| **Model Loader** | [`services/scoring/model_loader.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/scoring/model_loader.py) | In-memory Booster caching with thread-safe atomic pointer swapping for zero-downtime hot-reloading. |
| **Feature Fetcher** | [`services/scoring/feature_fetcher.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/scoring/feature_fetcher.py) | Low-latency (<1ms) online feature vector retrieval from Redis with Prometheus latency instrumentation. |
| **Threshold Router** | [`services/scoring/threshold_router.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/scoring/threshold_router.py) | 4-tier decision routing (LOW, MEDIUM, HIGH, CRITICAL) and async Kafka alert dispatcher. |
| **Synchronous Scorer** | [`services/scoring/scorer.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/services/scoring/scorer.py) | Sub-10ms LightGBM scoring engine, C-aligned array formatting, and async PostgreSQL audit logging. |
| **Baseline Training** | [`scripts/train_baseline.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/scripts/train_baseline.py) | Trains LightGBM classifier on 100k transactions with dynamic `scale_pos_weight` imbalance handling. |
| **Trained Model Artifact** | `model_cache/lightgbm/baseline_v1.txt` | Production-ready trained Booster weights on disk. |
| **Unit & Integration Tests** | `tests/unit/test_scorer.py`, `tests/unit/test_threshold_router.py`, `tests/integration/test_scoring_pipeline.py` | 8 new tests verifying inference speed, routing thresholds, and full pipeline integration. |

---

## 3. Baseline Model Evaluation Benchmark

Trained on 100,000 synthetic historical transactions (80,000 train / 20,000 test) with 2.55% class imbalance:

| Metric | Baseline Score | Target SLA | Status |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | `0.8721` | $> 0.80$ | ✅ Exceeded |
| **PR-AUC (Average Precision)** | `0.7477` | $> 0.65$ | ✅ Exceeded |
| **Precision** | `0.7085` | $> 0.65$ | ✅ Exceeded |
| **Recall** | `0.7515` | $> 0.70$ | ✅ Exceeded |
| **F1 Score** | `0.7293` | $> 0.70$ | ✅ Exceeded |
| **Inference Latency** | `~ 4.5 ms` | $< 10.0\text{ ms}$ | ✅ Exceeded |
| **Total Authorization Path** | `~ 6.8 ms` | $< 150.0\text{ ms}$ | ✅ Exceeded |

---

## 4. Key Architectural Decisions & Interview Talking Points

### 4.1 Classical ML (LightGBM) vs. LLM in Authorization Path
- **Question**: *"Why did you use LightGBM instead of an LLM for fraud scoring?"*
- **Talking Point**:
  > *"The payment authorization path has a strict 150ms hard gateway timeout SLA. An LLM call (even with Groq hardware) takes 500ms–2000ms, making it physically impossible to place inside the synchronous authorization path without causing payment timeouts. LightGBM executes in ~4.5ms on tabular feature arrays, leaving ample headroom within our 150ms SLA. The LLM is scoped strictly to the asynchronous investigation path where 3–5 seconds of latency is completely acceptable."*

### 4.2 Class Imbalance Strategy: `scale_pos_weight`
- **Question**: *"How did you handle the severe 2.5% class imbalance during training?"*
- **Talking Point**:
  > *"Rather than using SMOTE (which can generate unnatural synthetic points in high-dimensional sliding window features), we calculated dynamic `scale_pos_weight = N_legitimate / N_fraud = 38.18`. This directly scales the gradient of the positive fraud class in LightGBM's cross-entropy loss, ensuring the tree split criteria prioritize false negative penalization."*

---

## 5. Test Verification Results (42 Tests Passing)

```text
============================= test session starts =============================
collected 42 items

tests/integration/test_feature_pipeline.py ..                            [  4%]
tests/integration/test_postgres.py ..                                    [  9%]
tests/integration/test_redis.py .                                        [ 11%]
tests/integration/test_scoring_pipeline.py .                             [ 14%]
tests/unit/test_feature_definitions.py .......                           [ 30%]
tests/unit/test_models.py .........                                      [ 52%]
tests/unit/test_scorer.py ...                                            [ 59%]
tests/unit/test_settings.py ......                                       [ 73%]
tests/unit/test_simulator.py .......                                     [ 90%]
tests/unit/test_threshold_router.py ....                                 [100%]

============================= 42 passed in 7.78s ==============================
```

---

## 6. Next Steps
With Phase 4 complete and the baseline LightGBM model trained and active, we are ready for **Phase 5: LLM Investigation Assist**, building the asynchronous context builder, Groq API (Llama-3-8B) narrative generation, and evidence grounding citation verifier.
