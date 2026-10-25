"""
SentinelForge — ml/fine_tuning/dataset_curator.py

PURPOSE:
    Curates high-quality instruction-tuning datasets for Llama-3-8B fine-tuning.
    Transforms raw case dossiers and verified analyst adjudications into
    supervised instruction-input-output pairs with strict bracketed citations [Citing: key=value].

ARCHITECTURE POSITION:
    Layer 8 (LLM Fine-Tuning Pipeline) → Supervised Dataset Preparation.

INTERVIEW TALKING POINT:
    "We don't fine-tune on unstructured text. We curate strict Alpaca-formatted instruction pairs
    where every output narrative must score >= 0.85 on Evidence Grounding before inclusion.
    This teaches Llama-3-8B to internalize the domain grammar of financial fraud investigation
    while enforcing strict factual fidelity against the input context bundle."
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from services.investigation.evidence_grounding import EvidenceGroundingVerifier
from services.investigation.prompts import INVESTIGATION_SYSTEM_PROMPT
from shared.logging.logger import get_logger

logger = get_logger("ml.fine_tuning.curator")


class DatasetCurator:
    """
    Assembles, validates, and partitions supervised instruction datasets for QLoRA fine-tuning.
    """

    def __init__(self, verifier: Optional[EvidenceGroundingVerifier] = None) -> None:
        """
        Initialize DatasetCurator.

        Args:
            verifier: Grounding verifier used to validate output fidelity before inclusion.
        """
        self.verifier = verifier or EvidenceGroundingVerifier()

    def build_instruction_pair(
        self,
        case_id: str,
        context_bundle: Dict[str, Any],
        ground_truth_narrative: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a case context and narrative into an Alpaca-style instruction record.

        Args:
            case_id: Unique case identifier.
            context_bundle: Enriched 360-degree case context dictionary.
            ground_truth_narrative: Ground-truth narrative markdown with citations.

        Returns:
            Optional[Dict[str, Any]]: Validated training record or None if fidelity is insufficient.
        """
        # Validate that narrative meets grounding fidelity requirement (>= 0.85)
        score, verified, ungrounded, is_acceptable = self.verifier.verify_narrative(
            ground_truth_narrative, context_bundle
        )

        if score < 0.85 or not is_acceptable:
            logger.debug(
                "Skipping candidate narrative due to low grounding fidelity",
                case_id=case_id,
                score=score,
            )
            return None

        # Build Alpaca instruction record
        record = {
            "id": case_id,
            "instruction": INVESTIGATION_SYSTEM_PROMPT.strip(),
            "input": json.dumps(context_bundle, separators=(",", ":")),
            "output": ground_truth_narrative.strip(),
            "grounding_fidelity_score": round(score, 3),
            "citations_count": len(verified),
        }
        return record

    def curate_dataset_from_cases(
        self,
        cases_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Curate a list of raw case records into validated instruction pairs.

        Args:
            cases_list: List of dictionaries containing 'case_id', 'context_bundle', and 'narrative'.

        Returns:
            List[Dict[str, Any]]: Filtered list of verified instruction training records.
        """
        curated: List[Dict[str, Any]] = []

        for case in cases_list:
            case_id = case.get("case_id", f"case_{len(curated)}")
            context = case.get("context_bundle", {})
            narrative = case.get("narrative", "")

            pair = self.build_instruction_pair(case_id, context, narrative)
            if pair:
                curated.append(pair)

        logger.info(
            "Curated instruction dataset",
            total_candidates=len(cases_list),
            retained_high_fidelity=len(curated),
            retention_rate=round(len(curated) / max(1, len(cases_list)), 3),
        )
        return curated

    def partition_and_export(
        self,
        dataset: List[Dict[str, Any]],
        output_dir: str = "data/fine_tuning",
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        random_state: int = 42,
    ) -> Tuple[Path, Path, Path]:
        """
        Partition dataset into train/val/test splits and export to JSONL.

        Args:
            dataset: List of instruction training dictionaries.
            output_dir: Destination directory.
            train_ratio: Proportion of training examples.
            val_ratio: Proportion of validation examples.
            test_ratio: Proportion of held-out test examples.
            random_state: PRNG seed.

        Returns:
            Tuple[Path, Path, Path]: Paths to (train.jsonl, val.jsonl, test.jsonl).
        """
        if len(dataset) < 3:
            raise ValueError(f"Dataset must contain at least 3 examples to partition, got {len(dataset)}")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        train_path = out_path / "fraud_investigation_train.jsonl"
        val_path = out_path / "fraud_investigation_val.jsonl"
        test_path = out_path / "fraud_investigation_test.jsonl"

        # Split: Train vs (Val + Test)
        val_test_ratio = val_ratio + test_ratio
        train_data, val_test_data = train_test_split(
            dataset, test_size=val_test_ratio, random_state=random_state
        )

        # Split: Val vs Test
        val_data, test_data = train_test_split(
            val_test_data, test_size=0.5, random_state=random_state
        )

        for filepath, data in [
            (train_path, train_data),
            (val_path, val_data),
            (test_path, test_data),
        ]:
            with open(filepath, "w", encoding="utf-8") as f:
                for row in data:
                    f.write(json.dumps(row) + "\n")

        logger.info(
            "Exported fine-tuning JSONL splits",
            train_count=len(train_data),
            val_count=len(val_data),
            test_count=len(test_data),
            train_path=str(train_path),
        )
        return train_path, val_path, test_path
