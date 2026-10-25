"""
SentinelForge — ml/fine_tuning/export_adapter.py

PURPOSE:
    Validates, packages, and registers trained LoRA adapter weights for deployment.
    Generates deployment manifests and updates the local adapter registry for Layer 4/6 integration.

ARCHITECTURE POSITION:
    Layer 8 (LLM Fine-Tuning Pipeline) → Model Packaging & Registration.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from shared.logging.logger import get_logger

logger = get_logger("ml.fine_tuning.export")


def export_and_register_adapter(
    adapter_dir: str = "model_cache/lora_adapters/sentinel_llama3_v1",
    registry_file: str = "model_cache/lora_adapters/registry.json",
    adapter_name: str = "sentinel_llama3_investigator_v1",
    benchmark_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validate adapter directory and append entry to global registry.

    Args:
        adapter_dir: Directory containing adapter_config.json and weights.
        registry_file: JSON registry file tracking available LoRA adapters.
        adapter_name: Tag identifier.
        benchmark_metrics: Evaluation metrics (fidelity, ROUGE-L, etc.).

    Returns:
        Dict[str, Any]: Registered record.
    """
    ad_path = Path(adapter_dir)
    config_file = ad_path / "adapter_config.json"

    if not config_file.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_dir}")

    with open(config_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    reg_path = Path(registry_file)
    reg_path.parent.mkdir(parents=True, exist_ok=True)

    registry = {}
    if reg_path.exists():
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = {}

    record = {
        "adapter_name": adapter_name,
        "adapter_path": str(ad_path.resolve()),
        "base_model": config_data.get("base_model_name_or_path", "Meta-Llama-3-8B-Instruct"),
        "lora_r": config_data.get("r", 16),
        "lora_alpha": config_data.get("lora_alpha", 32),
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_DEPLOYMENT",
        "benchmark_metrics": benchmark_metrics or {},
    }

    registry[adapter_name] = record

    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    logger.info(
        "LoRA adapter registered successfully",
        adapter_name=adapter_name,
        registry_file=str(reg_path),
    )
    return record
