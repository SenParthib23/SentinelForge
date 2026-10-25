# Phase 7 Completion: Model Training Pipeline & Evaluation Harness

## 1. Overview & Objective
Phase 7 delivers the production-grade offline machine learning retraining and evaluation infrastructure of **SentinelForge**. It encapsulates:
1. **Layer 7 (Model Training Pipeline)**: Dynamic ingestion from historical Parquet and PostgreSQL analyst feedback (`labeled_data`), strict Single Source of Truth feature compilation, cost-sensitive class imbalance weighting, and cross-validated hyperparameter tuning.
2. **Layer 9 (Evaluation Harness)**: A rigorous **Two-Gate Promotion Architecture**. Gate 1 validates absolute baseline operational viability (AUC-ROC $\ge 0.85$, PR-AUC $\ge 0.70$, Recall@1% FPR $\ge 0.60$, Latency p99 $< 10\text{ms}$). Gate 2 executes the **Head-to-Head Regression Showdown** comparing Candidate Model ($M_{\text{cand}}$) directly against the active Production Champion ($M_{\text{prod}}$) on an identical held-out test distribution.

---

## 2. Artifacts & Deliverables Completed

| Component | Path | Description & Architectural Value |
| :--- | :--- | :--- |
| **Data Ingestion Engine** | [`ml/training/data_loader.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/ml/training/data_loader.py) | Ingests historical Parquet datasets, dynamically integrates fresh analyst ground-truth labels from PostgreSQL `labeled_data`, and executes stratified 3-way partitioning (70/15/15) preserving fraud distribution. |
| **Feature Matrix Builder** | [`ml/training/feature_builder.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/ml/training/feature_builder.py) | Transforms raw transactions into numerical matrices strictly ordered by `FEATURE_COLUMN_ORDER`, preventing training-serving skew. |
| **Class Imbalance & Cost Optimizer** | [`ml/training/class_imbalance.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/ml/training/class_imbalance.py) | Dynamically computes `scale_pos_weight = N_neg / N_pos` (~38.17) and scans decision thresholds to minimize financial loss ($C_{\text{FN}}=\$500$, $C_{\text{FP}}=\$10$). |
| **Hyperparameter Search** | [`ml/training/hyperparameter_search.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/ml/training/hyperparameter_search.py) | Stratified K-Fold cross-validation tuning tree depth, leaf count, subsampling, and learning rates specifically to maximize PR-AUC. |
| **Training Pipeline Orchestrator** | [`ml/training/train_lightgbm.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/ml/training/train_lightgbm.py) | End-to-end retraining script producing booster weights and comprehensive JSON training metadata. |
| **Domain Metrics Engine** | [`ml/evaluation/metrics.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/ml/evaluation/metrics.py) | Computes ROC-AUC, PR-AUC, F1, Recall@1% False Positive Rate, Precision@90% Recall, and inference latency percentiles (p50, p95, p99). |
| **Gate 1: Absolute Benchmark Gate** | [`ml/evaluation/benchmark_eval.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/ml/evaluation/benchmark_eval.py) | Gatekeeper verifying minimum operational criteria before candidates are eligible for regression testing. |
| **Gate 2: Head-to-Head Regression Suite** | [`ml/evaluation/regression_eval.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/ml/evaluation/regression_eval.py) | **The Core Differentiator**: Side-by-side comparison between Candidate and Production Champion on identical traffic slices. |
| **Audit Report Generator** | [`ml/evaluation/eval_report.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/ml/evaluation/eval_report.py) | Generates standardized Markdown and JSON audit reports for compliance and deployment gating. |
| **Evaluation CLI Runner** | [`scripts/run_regression_eval.py`](file:///d:/AI_ML_Bootcamp/2.%20SentinelForge/scripts/run_regression_eval.py) | Production CLI script running Gate 1 and Gate 2 evaluation and outputting reports. |
| **Test Suites** | `tests/unit/test_data_loader.py`, `test_class_imbalance.py`, `test_eval_metrics.py`, `test_regression_eval.py`, `test_training_pipeline.py` | 11 new tests verifying partitioning, loss weighting, metrics, regression gating, and training execution. |

---

## 3. Mathematical Formulations & Operational Principles

### 3.1 Dynamic Loss Weighting for 2.5% Fraud Imbalance
Standard binary cross-entropy minimizes loss by predicting zero on all samples. SentinelForge adjusts LightGBM's positive gradient weight dynamically:
$$\text{scale\_pos\_weight} = \frac{N_{\text{legitimate}}}{N_{\text{fraud}}} = \frac{68,213}{1,787} \approx 38.17$$
This gives gradient boosting trees equal loss sensitivity on sparse fraud events without introducing synthetic oversampling artifacts (e.g. SMOTE) into high-dimensional sliding window features.

### 3.2 Cost-Sensitive Threshold Discovery
Rather than arbitrarily defaulting to $0.50$, our validation scan discovers the operational threshold $\theta^*$ that minimizes the asymmetric banking cost function:
$$\text{Total Financial Cost} = C_{\text{FN}} \cdot \text{FN}(\theta) + C_{\text{FP}} \cdot \text{FP}(\theta)$$
where $C_{\text{FN}} = \$500$ (chargeback loss + dispute penalty) and $C_{\text{FP}} = \$10$ (cardholder friction / SMS review).

### 3.3 Payments Domain Metric: Recall@1% False Positive Rate
Banks enforce strict limits on false alarms to avoid declining good customers at checkout:
$$\text{Recall@1\% FPR} = \max_{\theta} \left\{ \text{TPR}(\theta) \mid \text{FPR}(\theta) \le 0.01 \right\}$$

---

## 4. Two-Gate Promotion Architecture in Action

During testing, we trained `candidate_v2` and executed `scripts/run_regression_eval.py` against active production champion `baseline_v1`:

```text
================================================================================
GATE 1 (Absolute Benchmark): PASSED
- AUC-ROC: 0.8858 (Required >= 0.8500) -> PASS
- PR-AUC: 0.7411 (Required >= 0.7000)  -> PASS
- Recall@1% FPR: 0.7598 (Required >= 0.6000) -> PASS
- Latency p99: 0.07 ms (Required < 10.0 ms)  -> PASS

GATE 2 (Head-to-Head Regression Showdown against baseline_v1): REJECTED
- Delta PR-AUC: -0.0443 (Candidate: 0.7411 vs Production: 0.7854) -> REGRESSED
- Delta AUC-ROC: -0.0229 (Candidate: 0.8858 vs Production: 0.9087) -> REGRESSED
- Delta Recall@1% FPR: -0.0261 (Candidate: 0.7598 vs Production: 0.7859) -> REGRESSED

VERDICT: PROMOTION BLOCKED
Automated Canary Rollout Permitted: NO
================================================================================
```

### Key Architectural Insight:
A naive ML platform checking only static benchmarks would have prematurely promoted `candidate_v2` because it scored $>0.88$ AUC. SentinelForge's **Gate 2 Showdown caught that the candidate degraded relative to the active production model**, successfully blocking deployment and preserving risk operations stability.

---

## 5. Test Suite Verification (69 Tests Passing)

```text
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
collected 69 items

tests/integration/test_feature_pipeline.py ..                            [  2%]
tests/integration/test_investigation_pipeline.py .                       [  4%]
tests/integration/test_postgres.py ..                                    [  7%]
tests/integration/test_redis.py .                                        [  8%]
tests/integration/test_scoring_pipeline.py .                             [ 10%]
tests/integration/test_training_pipeline.py .                            [ 11%]
tests/unit/test_api_auth.py ...                                          [ 15%]
tests/unit/test_api_health_and_models.py ...                             [ 20%]
tests/unit/test_api_investigation.py ...                                 [ 24%]
tests/unit/test_api_scoring.py .                                         [ 26%]
tests/unit/test_case_builder.py ..                                       [ 28%]
tests/unit/test_class_imbalance.py ...                                   [ 33%]
tests/unit/test_data_loader.py ..                                        [ 36%]
tests/unit/test_eval_metrics.py ...                                      [ 40%]
tests/unit/test_evidence_grounding.py ...                                [ 44%]
tests/unit/test_feature_definitions.py .......                           [ 55%]
tests/unit/test_models.py .........                                      [ 68%]
tests/unit/test_regression_eval.py ..                                    [ 71%]
tests/unit/test_scorer.py ...                                            [ 75%]
tests/unit/test_settings.py ......                                       [ 84%]
tests/unit/test_simulator.py .......                                     [ 94%]
tests/unit/test_threshold_router.py ....                                 [100%]

============================= 69 passed in 14.18s =============================
```

---

## 6. Next Steps
With the ML training pipeline and evaluation regression gates verified, we are ready for **Phase 8: LLM Fine-Tuning Pipeline (QLoRA)**:
- Curating domain-specific fraud investigation instruction pairs from analyst-reviewed cases.
- Low-Rank Adaptation (LoRA) fine-tuning scripts for Llama-3-8B.
- ROUGE-L, grounding fidelity, and factual consistency evaluation.
