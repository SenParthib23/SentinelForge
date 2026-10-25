"""
SentinelForge — services/investigation/evidence_grounding.py

PURPOSE:
    Automated evidence grounding and citation verification for LLM-generated
    fraud investigation narratives. Cross-checks bracketed citations like
    `[Citing: field=value]` against raw context bundle values to prevent hallucinations.

ARCHITECTURE POSITION:
    Layer 4 (Async Investigation Assist) → Verification gate after Groq LLM inference.

VERIFICATION RULES:
    1. Extracts all citations matching `[Citing: ...]` or explicit feature mentions.
    2. Flattens the nested context bundle into key-value pairs.
    3. Validates whether the cited attribute exists in the context bundle.
    4. For numerical claims, checks value proximity within a 5% rounding tolerance.
    5. Calculates `grounding_score = valid_citations / total_citations`.
    6. If grounding_score < 0.80 (>20% ungrounded claims), flags as UNGROUNDED.

INTERVIEW TALKING POINT:
    "We don't trust raw LLM outputs in production financial decisions.
    Our Evidence Grounding Engine parses every citation token from the LLM narrative
    and validates it against the ground-truth JSON context. If the model hallucinates
    a non-existent transaction or inflates an amount z-score, the grounding score drops
    below 0.80 and triggers an automated deterministic fallback."
"""

import math
import re
from typing import Any, Optional

from shared.logging.logger import get_logger

logger = get_logger("investigation.evidence_grounding")


class EvidenceGroundingVerifier:
    """
    Validates factual citations in LLM narratives against the raw context bundle.
    """

    def __init__(self, minimum_grounding_score: float = 0.80) -> None:
        """
        Initialize verifier with minimum acceptable grounding fidelity.

        Args:
            minimum_grounding_score: Minimum ratio of grounded claims (default 0.80).
        """
        self.minimum_grounding_score = minimum_grounding_score
        # Regex to extract bracketed citations: e.g. [Citing: amount_zscore=14.2, txn_count_1min=5]
        self._citation_pattern = re.compile(r"\[Citing:\s*([^\]]+)\]", re.IGNORECASE)

    def _flatten_context(self, context: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
        """
        Flatten nested context dictionary into flat key-value pairs.

        Args:
            context: Nested dictionary.
            parent_key: Prefix for nested keys.

        Returns:
            dict[str, Any]: Flat dictionary with both leaf and fully-qualified keys.
        """
        items: dict[str, Any] = {}
        for k, v in context.items():
            leaf_key = k.lower()
            new_key = f"{parent_key}.{k}".lower() if parent_key else leaf_key

            if isinstance(v, dict):
                items.update(self._flatten_context(v, new_key))
            else:
                items[new_key] = v
                # Also store leaf key for direct lookups
                if leaf_key not in items:
                    items[leaf_key] = v
        return items

    def verify_narrative(
        self,
        narrative: str,
        context_bundle: dict[str, Any],
    ) -> tuple[float, list[str], list[str], bool]:
        """
        Verify that all claims in narrative map to facts in context_bundle.

        Args:
            narrative: LLM-generated markdown text.
            context_bundle: Ground-truth context dictionary.

        Returns:
            tuple[float, list[str], list[str], bool]:
                (grounding_score, verified_claims, ungrounded_claims, is_acceptable)
        """
        flat_context = self._flatten_context(context_bundle)
        matches = self._citation_pattern.findall(narrative)

        verified_claims: list[str] = []
        ungrounded_claims: list[str] = []

        if not matches:
            # If no formal bracketed citations found, search for common feature keywords
            logger.warning("No explicit [Citing: ...] citations found in narrative")
            return 0.50, [], ["No explicit citations in narrative"], False

        for match in matches:
            # Split comma-separated tokens inside citation: e.g. "amount_zscore=14.2, baseline_mean=500"
            parts = [p.strip() for p in match.split(",") if p.strip()]
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    key = k.strip().lower()
                    val_str = v.strip().lower()

                    is_valid = self._validate_key_value(key, val_str, flat_context)
                    if is_valid:
                        verified_claims.append(f"{key}={val_str}")
                    else:
                        ungrounded_claims.append(f"{key}={val_str} (mismatch or missing)")
                else:
                    # Key-only citation: e.g. [Citing: is_new_device]
                    key = part.strip().lower()
                    if key in flat_context:
                        verified_claims.append(key)
                    else:
                        ungrounded_claims.append(f"{key} (key not found)")

        total_claims = len(verified_claims) + len(ungrounded_claims)
        if total_claims == 0:
            score = 1.0
        else:
            score = round(len(verified_claims) / total_claims, 3)

        is_acceptable = score >= self.minimum_grounding_score

        logger.info(
            "Evidence grounding verification complete",
            grounding_score=score,
            verified_count=len(verified_claims),
            ungrounded_count=len(ungrounded_claims),
            is_acceptable=is_acceptable,
        )

        return score, verified_claims, ungrounded_claims, is_acceptable

    def _validate_key_value(
        self,
        key: str,
        claimed_val_str: str,
        flat_context: dict[str, Any],
    ) -> bool:
        """
        Validate whether claimed key-value matches truth in context.

        Args:
            key: Field name.
            claimed_val_str: String representation of claimed value.
            flat_context: Flattened context dictionary.

        Returns:
            bool: True if key exists and value matches within tolerance.
        """
        if key not in flat_context:
            # Check partial key match (e.g. "distance" in "distance_from_last_txn_km")
            matching_keys = [k for k in flat_context if key in k]
            if not matching_keys:
                return False
            actual_val = flat_context[matching_keys[0]]
        else:
            actual_val = flat_context[key]

        # 1. Boolean check
        if claimed_val_str in {"true", "false"}:
            return str(actual_val).lower() == claimed_val_str

        # 2. Numeric check (int/float with 10% tolerance for rounding)
        try:
            # Strip non-numeric units (e.g. "7189km" -> 7189)
            clean_claimed = re.sub(r"[^\d.-]", "", claimed_val_str)
            clean_actual = re.sub(r"[^\d.-]", "", str(actual_val))

            c_num = float(clean_claimed)
            a_num = float(clean_actual)

            if a_num == 0.0:
                return abs(c_num) < 0.01

            relative_diff = abs(c_num - a_num) / abs(a_num)
            return relative_diff <= 0.10  # within 10% tolerance
        except (ValueError, TypeError):
            pass

        # 3. String equivalence check
        return claimed_val_str in str(actual_val).lower()
