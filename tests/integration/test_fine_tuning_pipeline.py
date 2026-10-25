"""
SentinelForge — tests/integration/test_fine_tuning_pipeline.py

PURPOSE:
    Integration test verifying complete QLoRA training run, adapter config export,
    and registry registration.
"""

import json
from pathlib import Path

import pytest

from ml.fine_tuning.export_adapter import export_and_register_adapter
from ml.fine_tuning.qlora_config import QLoRATrainingConfig
from ml.fine_tuning.train_qlora import train_qlora


def test_fine_tuning_pipeline_and_registry(tmp_path: Path) -> None:
    """Verify train_qlora generates valid adapter files and registry record."""
    data_file = tmp_path / "train.jsonl"
    adapter_dir = tmp_path / "adapter_export"
    registry_file = tmp_path / "registry.json"

    # 1. Create a dummy jsonl sample
    sample = {
        "id": "c1",
        "instruction": "System prompt",
        "input": "{}",
        "output": "Narrative",
    }
    with open(data_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample) + "\n")

    # 2. Run train_qlora in dry_run mode
    cfg = QLoRATrainingConfig(
        adapter_name="test_sentinel_lora",
        output_dir=str(adapter_dir),
        num_train_epochs=1,
    )
    summary = train_qlora(
        dataset_path=str(data_file),
        config=cfg,
        output_dir=str(adapter_dir),
        dry_run=True,
    )

    assert summary["status"] == "COMPLETED"
    assert summary["model_name"] == "test_sentinel_lora"

    # 3. Verify created files
    assert (adapter_dir / "adapter_config.json").exists()
    assert (adapter_dir / "tokenizer_config.json").exists()
    assert (adapter_dir / "training_metrics.json").exists()

    # 4. Register adapter
    record = export_and_register_adapter(
        adapter_dir=str(adapter_dir),
        registry_file=str(registry_file),
        adapter_name="test_sentinel_lora",
        benchmark_metrics={"fidelity": 0.95},
    )

    assert record["adapter_name"] == "test_sentinel_lora"
    assert record["status"] == "READY_FOR_DEPLOYMENT"

    with open(registry_file, "r", encoding="utf-8") as f:
        reg_data = json.load(f)
    assert "test_sentinel_lora" in reg_data
