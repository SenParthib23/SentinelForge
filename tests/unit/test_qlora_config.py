"""
SentinelForge — tests/unit/test_qlora_config.py

PURPOSE:
    Unit tests for QLoRATrainingConfig hyperparameter specifications and VRAM estimations.
"""

from pathlib import Path

import pytest

from ml.fine_tuning.qlora_config import QLoRATrainingConfig


def test_qlora_config_parameters_and_vram() -> None:
    """Verify QLoRA configuration parameters and theoretical memory reduction."""
    cfg = QLoRATrainingConfig(lora_r=16, lora_alpha=32)

    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert "q_proj" in cfg.target_modules
    assert "v_proj" in cfg.target_modules

    stats = cfg.calculate_parameter_stats()
    assert stats["trainable_percentage"] < 1.0  # < 1% trainable params
    assert stats["estimated_base_vram_gb_nf4"] < 6.0  # ~5.4 GB


def test_qlora_config_serialization(tmp_path: Path) -> None:
    """Verify configuration can be exported and reloaded as valid JSON."""
    cfg = QLoRATrainingConfig(adapter_name="test_adapter")
    save_file = tmp_path / "test_config.json"
    cfg.save(str(save_file))

    assert save_file.exists()
    d = cfg.to_dict()
    assert d["adapter_name"] == "test_adapter"
    assert d["lora_r"] == 16
