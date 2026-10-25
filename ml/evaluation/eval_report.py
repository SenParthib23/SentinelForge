"""
SentinelForge — ml/evaluation/eval_report.py

PURPOSE:
    Formats evaluation outcomes into standardized human-readable Markdown and
    structured JSON artifacts for audit trails, MLOps dashboards, and deployment gating.

ARCHITECTURE POSITION:
    Layer 9 (Evaluation Harness) → Report Generation.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from ml.evaluation.benchmark_eval import BenchmarkResult
from ml.evaluation.regression_eval import RegressionResult


def generate_evaluation_report(
    candidate_name: str,
    production_name: str,
    benchmark_res: BenchmarkResult,
    regression_res: RegressionResult,
    output_path: Optional[str] = None,
) -> str:
    """
    Format Gate 1 and Gate 2 results into comprehensive Markdown audit report.

    Args:
        candidate_name: Candidate model tag (e.g. 'candidate_v2').
        production_name: Active production champion tag (e.g. 'baseline_v1').
        benchmark_res: BenchmarkResult from Gate 1.
        regression_res: RegressionResult from Gate 2.
        output_path: Optional file path to persist Markdown report.

    Returns:
        str: Formatted Markdown report string.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    overall_status = "APPROVED FOR PROMOTION" if (benchmark_res.passed and regression_res.is_promotable) else "PROMOTION BLOCKED"
    status_icon = "🟢" if overall_status == "APPROVED FOR PROMOTION" else "🔴"

    cand_m = regression_res.candidate_metrics
    prod_m = regression_res.production_metrics
    deltas = regression_res.deltas

    md = f"""# SentinelForge Model Evaluation & Promotion Audit Report

**Generated At:** {now}  
**Candidate Model:** `{candidate_name}`  
**Current Production Champion:** `{production_name}`  
**Overall Verdict:** {status_icon} **{overall_status}**

---

## 1. Gate 1: Absolute Operational Benchmark

| Metric | Measured Value | Minimum Required | Gate Status |
| :--- | :--- | :--- | :--- |
| **AUC-ROC** | `{benchmark_res.metrics['auc_roc']:.4f}` | `0.8500` | {"PASS" if benchmark_res.metrics['auc_roc'] >= 0.85 else "FAIL"} |
| **PR-AUC** | `{benchmark_res.metrics['pr_auc']:.4f}` | `0.7000` | {"PASS" if benchmark_res.metrics['pr_auc'] >= 0.70 else "FAIL"} |
| **Recall@1% FPR** | `{benchmark_res.metrics['recall_at_1_fpr']:.4f}` | `0.6000` | {"PASS" if benchmark_res.metrics['recall_at_1_fpr'] >= 0.60 else "FAIL"} |
| **Inference Latency (p99)** | `{benchmark_res.metrics['latency_p99_ms']:.2f} ms` | `< 10.00 ms` | {"PASS" if benchmark_res.metrics['latency_p99_ms'] <= 10.0 else "FAIL"} |

*Gate 1 Overall: **{"PASSED" if benchmark_res.passed else "FAILED"}***
{f"> Violations: {', '.join(benchmark_res.violations)}" if benchmark_res.violations else ""}

---

## 2. Gate 2: Head-to-Head Regression Showdown (Held-Out Test Slice)

| Metric | Production Champion (`{production_name}`) | Candidate (`{candidate_name}`) | Delta (Cand - Prod) | Regression Check |
| :--- | :--- | :--- | :--- | :--- |
| **PR-AUC** | `{prod_m['pr_auc']:.4f}` | `{cand_m['pr_auc']:.4f}` | `{deltas['delta_pr_auc']:+.4f}` | {"PASS" if deltas['delta_pr_auc'] >= 0 else "REGRESSED"} |
| **AUC-ROC** | `{prod_m['auc_roc']:.4f}` | `{cand_m['auc_roc']:.4f}` | `{deltas['delta_auc_roc']:+.4f}` | {"PASS" if deltas['delta_auc_roc'] >= -0.005 else "REGRESSED"} |
| **Recall@1% FPR** | `{prod_m['recall_at_1_fpr']:.4f}` | `{cand_m['recall_at_1_fpr']:.4f}` | `{deltas['delta_recall_at_1_fpr']:+.4f}` | {"PASS" if deltas['delta_recall_at_1_fpr'] >= 0 else "REGRESSED"} |
| **Precision@90% Recall**| `{prod_m['precision_at_90_recall']:.4f}` | `{cand_m['precision_at_90_recall']:.4f}` | `{cand_m['precision_at_90_recall'] - prod_m['precision_at_90_recall']:+.4f}` | Informational |
| **F1 Score** | `{prod_m['f1_score']:.4f}` | `{cand_m['f1_score']:.4f}` | `{deltas['delta_f1_score']:+.4f}` | Informational |
| **Latency p99** | `{prod_m['latency_p99_ms']:.2f} ms` | `{cand_m['latency_p99_ms']:.2f} ms` | `{deltas['delta_latency_p99_ms']:+.2f} ms` | {"PASS" if deltas['delta_latency_p99_ms'] <= 1.0 else "REGRESSED"} |

*Gate 2 Showdown: **{"APPROVED" if regression_res.is_promotable else "BLOCKED"}***
{f"> Regressions Detected: {', '.join(regression_res.regression_reasons)}" if regression_res.regression_reasons else ""}

---

## 3. Deployment Recommendation

```
Decision: {overall_status}
Automated Canary Rollout Permitted: {'YES' if overall_status == 'APPROVED FOR PROMOTION' else 'NO'}
```
"""
    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(md)

    return md
