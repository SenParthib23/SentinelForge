"""
SentinelForge — tests/unit/test_dataset_curator.py

PURPOSE:
    Unit tests for DatasetCurator instruction-tuning pair generation and partitioning.
"""

import json
from pathlib import Path

import pytest

from ml.fine_tuning.dataset_curator import DatasetCurator


def test_build_instruction_pair_valid_grounding() -> None:
    """Verify valid grounded narrative is transformed into Alpaca instruction pair."""
    curator = DatasetCurator()

    ctx = {
        "transaction_info": {"amount": 7500.0, "currency": "INR"},
        "feature_snapshot": {"amount_zscore": 3.9},
    }
    narrative = """## Transaction Summary
Flagged transaction.

## Risk Indicators
- Amount of INR 7500.0 is high [Citing: amount=7500.0, amount_zscore=3.9].

## Customer Context & Identity Graph
Baseline spend review.

## Recommended Action
HOLD_FOR_REVIEW
Verify with customer."""

    pair = curator.build_instruction_pair("case_test_01", ctx, narrative)
    assert pair is not None
    assert pair["id"] == "case_test_01"
    assert "instruction" in pair
    assert "input" in pair
    assert "output" in pair
    assert pair["grounding_fidelity_score"] >= 0.85
    assert pair["citations_count"] >= 1


def test_build_instruction_pair_rejects_hallucinations() -> None:
    """Verify narrative with ungrounded hallucinations is excluded."""
    curator = DatasetCurator()

    ctx = {"transaction_info": {"amount": 100.0}}
    # Claims completely fake amount 99999.0
    fake_narrative = "Transaction of 99999.0 was flagged [Citing: amount=99999.0]."

    pair = curator.build_instruction_pair("case_test_02", ctx, fake_narrative)
    assert pair is None


def test_partition_and_export(tmp_path: Path) -> None:
    """Verify partitioning creates train, val, and test JSONL files."""
    curator = DatasetCurator()

    records = [
        {
            "id": f"c_{i}",
            "instruction": "Instruct",
            "input": "{}",
            "output": f"Output {i}",
            "grounding_fidelity_score": 1.0,
            "citations_count": 2,
        }
        for i in range(10)
    ]

    train_p, val_p, test_p = curator.partition_and_export(
        records, output_dir=str(tmp_path), train_ratio=0.60, val_ratio=0.20, test_ratio=0.20
    )

    assert train_p.exists()
    assert val_p.exists()
    assert test_p.exists()

    with open(train_p, "r", encoding="utf-8") as f:
        train_lines = f.readlines()
    assert len(train_lines) == 6
