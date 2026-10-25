"""
SentinelForge — ml/fine_tuning/eval_llm.py

PURPOSE:
    Multi-dimensional evaluation engine for fine-tuned LLM investigation agents.
    Evaluates:
    1. Evidence Grounding Fidelity & Hallucination Rate
    2. ROUGE-1, ROUGE-2, and ROUGE-L lexical overlap against human analyst ground truth
    3. Regulatory SAR (Suspicious Activity Report) compliance and structural completeness.

ARCHITECTURE POSITION:
    Layer 8 (LLM Fine-Tuning Pipeline) → Fine-Tuned Model Evaluation.

INTERVIEW TALKING POINT:
    "Evaluating an LLM for fraud investigation cannot rely purely on NLP metrics like ROUGE or BLEU.
    A high ROUGE score is meaningless if the LLM invents a non-existent transaction amount.
    SentinelForge evaluates along three orthogonal dimensions: Grounding Fidelity (is every claim
    backed by real context?), ROUGE-L (does the tone and style match Senior Analysts?), and
    Regulatory SAR Compliance (are all 4 legally mandated investigation sections present?)."
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from services.investigation.evidence_grounding import EvidenceGroundingVerifier
from shared.logging.logger import get_logger

logger = get_logger("ml.fine_tuning.eval_llm")


@dataclass
class LLMEvalMetrics:
    """Summary metrics of LLM narrative evaluation."""
    grounding_fidelity_score: float
    hallucination_rate: float
    rouge_1_f1: float
    rouge_2_f1: float
    rouge_l_f1: float
    sar_compliance_rate: float
    total_evaluated: int
    passed_operational_gate: bool
    diagnostic_details: List[Dict[str, Any]] = field(default_factory=list)


class LLMEvaluator:
    """
    Evaluates generated investigation narratives for fidelity, style, and regulatory compliance.
    """

    def __init__(self, verifier: Optional[EvidenceGroundingVerifier] = None) -> None:
        """
        Initialize LLMEvaluator.

        Args:
            verifier: Grounding verifier for citation extraction and proximity checking.
        """
        self.verifier = verifier or EvidenceGroundingVerifier()

    def evaluate_narrative(
        self,
        generated_narrative: str,
        ground_truth_narrative: str,
        context_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate a single generated narrative against context and analyst ground truth.

        Args:
            generated_narrative: Output text from candidate fine-tuned LLM.
            ground_truth_narrative: Gold-standard narrative drafted by human analyst.
            context_bundle: Original 360-degree case context dictionary.

        Returns:
            Dict[str, Any]: Individual evaluation breakdown.
        """
        # 1. Evidence Grounding & Hallucinations
        fidelity, verified, ungrounded, is_acceptable = self.verifier.verify_narrative(
            generated_narrative, context_bundle
        )
        total_citations = len(verified) + len(ungrounded)
        hallucination_rate = (len(ungrounded) / total_citations) if total_citations > 0 else 0.0

        # 2. ROUGE Lexical Overlap
        r1 = self._compute_rouge_n(generated_narrative, ground_truth_narrative, n=1)
        r2 = self._compute_rouge_n(generated_narrative, ground_truth_narrative, n=2)
        rl = self._compute_rouge_l(generated_narrative, ground_truth_narrative)

        # 3. Regulatory SAR Compliance Check
        sar_compliant, missing_sections = self._check_sar_compliance(generated_narrative)

        return {
            "fidelity_score": round(fidelity, 4),
            "hallucination_rate": round(hallucination_rate, 4),
            "rouge_1": round(r1, 4),
            "rouge_2": round(r2, 4),
            "rouge_l": round(rl, 4),
            "sar_compliant": sar_compliant,
            "missing_sections": missing_sections,
            "verified_citations": len(verified),
            "ungrounded_citations": len(ungrounded),
        }

    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
        min_fidelity: float = 0.80,
        max_hallucination_rate: float = 0.15,
        min_rouge_l: float = 0.40,
        min_sar_compliance: float = 0.90,
    ) -> LLMEvalMetrics:
        """
        Evaluate a batch of generated narratives and assess operational promotion gate.

        Args:
            samples: List of dictionaries with 'generated', 'ground_truth', and 'context'.
            min_fidelity: Minimum acceptable average grounding fidelity (0.80).
            max_hallucination_rate: Maximum allowable hallucination rate (0.15).
            min_rouge_l: Minimum style similarity to analyst text (0.40).
            min_sar_compliance: Minimum fraction passing regulatory structure (0.90).

        Returns:
            LLMEvalMetrics: Consolidated benchmark score.
        """
        if not samples:
            return LLMEvalMetrics(
                grounding_fidelity_score=0.0,
                hallucination_rate=0.0,
                rouge_1_f1=0.0,
                rouge_2_f1=0.0,
                rouge_l_f1=0.0,
                sar_compliance_rate=0.0,
                total_evaluated=0,
                passed_operational_gate=False,
            )

        fidelities, hallucinations, r1s, r2s, rls, compliances = [], [], [], [], [], []
        details = []

        for s in samples:
            res = self.evaluate_narrative(
                generated_narrative=s["generated"],
                ground_truth_narrative=s["ground_truth"],
                context_bundle=s["context"],
            )
            fidelities.append(res["fidelity_score"])
            hallucinations.append(res["hallucination_rate"])
            r1s.append(res["rouge_1"])
            r2s.append(res["rouge_2"])
            rls.append(res["rouge_l"])
            compliances.append(1.0 if res["sar_compliant"] else 0.0)
            details.append(res)

        avg_fid = float(np.mean(fidelities))
        avg_halluc = float(np.mean(hallucinations))
        avg_r1 = float(np.mean(r1s))
        avg_r2 = float(np.mean(r2s))
        avg_rl = float(np.mean(rls))
        sar_rate = float(np.mean(compliances))

        passed = (
            avg_fid >= min_fidelity
            and avg_halluc <= max_hallucination_rate
            and avg_rl >= min_rouge_l
            and sar_rate >= min_sar_compliance
        )

        logger.info(
            "LLM Fine-Tuning Evaluation Complete",
            fidelity=round(avg_fid, 3),
            hallucination_rate=round(avg_halluc, 3),
            rouge_l=round(avg_rl, 3),
            sar_compliance=round(sar_rate, 3),
            passed_gate=passed,
        )

        return LLMEvalMetrics(
            grounding_fidelity_score=round(avg_fid, 4),
            hallucination_rate=round(avg_halluc, 4),
            rouge_1_f1=round(avg_r1, 4),
            rouge_2_f1=round(avg_r2, 4),
            rouge_l_f1=round(avg_rl, 4),
            sar_compliance_rate=round(sar_rate, 4),
            total_evaluated=len(samples),
            passed_operational_gate=passed,
            diagnostic_details=details,
        )

    def _check_sar_compliance(self, text: str) -> Tuple[bool, List[str]]:
        """
        Verify presence of mandatory regulatory SAR brief sections.
        """
        required = [
            ("Transaction Summary", r"##\s*transaction\s*summary"),
            ("Risk Indicators", r"##\s*risk\s*indicators"),
            ("Customer Context & Graph", r"##\s*customer\s*context"),
            ("Recommended Action", r"##\s*recommended\s*action"),
        ]
        missing = []
        for name, pattern in required:
            if not re.search(pattern, text, re.IGNORECASE):
                missing.append(name)

        has_disposition = any(
            d in text for d in ["APPROVE", "HOLD_FOR_REVIEW", "ESCALATE_URGENT"]
        )
        if not has_disposition:
            missing.append("Standard Disposition Tag (APPROVE/HOLD/ESCALATE)")

        return len(missing) == 0, missing

    def _compute_rouge_n(self, cand: str, ref: str, n: int = 1) -> float:
        """Compute ROUGE-N F1 score."""
        cand_tokens = self._tokenize(cand)
        ref_tokens = self._tokenize(ref)

        if len(cand_tokens) < n or len(ref_tokens) < n:
            return 0.0

        cand_ngrams = [tuple(cand_tokens[i:i+n]) for i in range(len(cand_tokens) - n + 1)]
        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]

        cand_set = set(cand_ngrams)
        ref_set = set(ref_ngrams)

        overlap = len(cand_set.intersection(ref_set))
        if overlap == 0:
            return 0.0

        prec = overlap / len(cand_set)
        rec = overlap / len(ref_set)
        return (2 * prec * rec) / (prec + rec)

    def _compute_rouge_l(self, cand: str, ref: str) -> float:
        """Compute ROUGE-L (Longest Common Subsequence) F1 score."""
        cand_tokens = self._tokenize(cand)
        ref_tokens = self._tokenize(ref)

        if not cand_tokens or not ref_tokens:
            return 0.0

        lcs_len = self._lcs_length(cand_tokens, ref_tokens)
        if lcs_len == 0:
            return 0.0

        prec = lcs_len / len(cand_tokens)
        rec = lcs_len / len(ref_tokens)
        return (2 * prec * rec) / (prec + rec)

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute length of longest common subsequence via dynamic programming."""
        m, n = len(seq1), len(seq2)
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            prev = 0
            for j in range(1, n + 1):
                temp = dp[j]
                if seq1[i - 1] == seq2[j - 1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = temp
        return dp[n]

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace and punctuation word tokenizer."""
        return re.findall(r"\b\w+\b", text.lower())
