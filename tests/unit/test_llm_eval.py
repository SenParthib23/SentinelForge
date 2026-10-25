"""
SentinelForge — tests/unit/test_llm_eval.py

PURPOSE:
    Unit tests for LLMEvaluator Grounding Fidelity, ROUGE overlap, and SAR compliance.
"""

import pytest

from ml.fine_tuning.eval_llm import LLMEvaluator


def test_evaluate_single_narrative() -> None:
    """Verify evaluation breakdown of grounded, compliant narrative."""
    evaluator = LLMEvaluator()

    ctx = {
        "transaction_info": {"amount": 5000.0, "currency": "INR"},
        "feature_snapshot": {"amount_zscore": 3.5},
    }
    narrative = """## Transaction Summary
Flagged transaction for INR 5000.00.

## Risk Indicators
- Amount of INR 5000.0 exceeds threshold [Citing: amount=5000.0, amount_zscore=3.5].

## Customer Context & Identity Graph
Baseline spend review.

## Recommended Action
HOLD_FOR_REVIEW
Hold for manual review."""

    res = evaluator.evaluate_narrative(
        generated_narrative=narrative,
        ground_truth_narrative=narrative,
        context_bundle=ctx,
    )

    assert res["fidelity_score"] >= 0.80
    assert res["hallucination_rate"] == 0.0
    assert res["sar_compliant"] is True
    assert res["rouge_l"] == 1.0  # Perfect match with ground truth


def test_sar_compliance_failure() -> None:
    """Verify missing sections are caught by compliance checker."""
    evaluator = LLMEvaluator()

    incomplete_text = "Just a short comment without sections or disposition."
    compliant, missing = evaluator._check_sar_compliance(incomplete_text)

    assert compliant is False
    assert len(missing) >= 3


def test_evaluate_batch_gate_pass() -> None:
    """Verify batch evaluation computes summary metrics and gate status."""
    evaluator = LLMEvaluator()

    ctx = {"transaction_info": {"amount": 1000.0}}
    text = """## Transaction Summary
Summary [Citing: amount=1000.0].
## Risk Indicators
Indicators.
## Customer Context & Identity Graph
Context.
## Recommended Action
APPROVE
Approved."""

    samples = [
        {"generated": text, "ground_truth": text, "context": ctx}
        for _ in range(3)
    ]

    metrics = evaluator.evaluate_batch(samples)
    assert metrics.total_evaluated == 3
    assert metrics.grounding_fidelity_score == 1.0
    assert metrics.sar_compliance_rate == 1.0
    assert metrics.passed_operational_gate is True
