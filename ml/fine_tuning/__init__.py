"""
SentinelForge — ml/fine_tuning package
Provides dataset curation, 4-bit QLoRA fine-tuning pipelines for Llama-3-8B,
and LLM evaluation suites (Grounding Fidelity, ROUGE-L, and SAR Compliance).
"""

from ml.fine_tuning.dataset_curator import DatasetCurator
from ml.fine_tuning.qlora_config import QLoRATrainingConfig

__all__ = ["DatasetCurator", "QLoRATrainingConfig"]
