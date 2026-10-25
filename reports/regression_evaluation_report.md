# SentinelForge Model Evaluation & Promotion Audit Report

**Generated At:** 2026-10-25 08:04:16 UTC  
**Candidate Model:** `candidate_v2`  
**Current Production Champion:** `baseline_v1`  
**Overall Verdict:** 🔴 **PROMOTION BLOCKED**

---

## 1. Gate 1: Absolute Operational Benchmark

| Metric | Measured Value | Minimum Required | Gate Status |
| :--- | :--- | :--- | :--- |
| **AUC-ROC** | `0.8858` | `0.8500` | PASS |
| **PR-AUC** | `0.7411` | `0.7000` | PASS |
| **Recall@1% FPR** | `0.7598` | `0.6000` | PASS |
| **Inference Latency (p99)** | `0.07 ms` | `< 10.00 ms` | PASS |

*Gate 1 Overall: **PASSED***


---

## 2. Gate 2: Head-to-Head Regression Showdown (Held-Out Test Slice)

| Metric | Production Champion (`baseline_v1`) | Candidate (`candidate_v2`) | Delta (Cand - Prod) | Regression Check |
| :--- | :--- | :--- | :--- | :--- |
| **PR-AUC** | `0.7854` | `0.7411` | `-0.0443` | REGRESSED |
| **AUC-ROC** | `0.9087` | `0.8858` | `-0.0229` | REGRESSED |
| **Recall@1% FPR** | `0.7859` | `0.7598` | `-0.0261` | REGRESSED |
| **Precision@90% Recall**| `0.0553` | `0.0304` | `-0.0249` | Informational |
| **F1 Score** | `0.7315` | `0.0000` | `-0.7315` | Informational |
| **Latency p99** | `0.13 ms` | `0.07 ms` | `-0.06 ms` | PASS |

*Gate 2 Showdown: **BLOCKED***
> Regressions Detected: PR-AUC regressed by -0.0443 (Cand: 0.7411 vs Prod: 0.7854), AUC-ROC dropped by -0.0229 (Cand: 0.8858 vs Prod: 0.9087), Recall@1% FPR dropped by -0.0261 (Cand: 0.7598 vs Prod: 0.7859)

---

## 3. Deployment Recommendation

```
Decision: PROMOTION BLOCKED
Automated Canary Rollout Permitted: NO
```
