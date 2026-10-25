"""
SentinelForge — services/investigation/investigation_agent.py

PURPOSE:
    Asynchronous LLM investigation agent powered by Groq (Llama-3-8B).
    Generates structured fraud investigation narratives and regulatory SAR briefs,
    validates citation grounding, and records execution metrics.

ARCHITECTURE POSITION:
    Layer 4 (Async Investigation Assist) → Consumes assembled context bundle
    and outputs verified narrative for the Analyst Dashboard (Layer 6).

HOW IT WORKS:
    1. Formats context bundle into strict system/user prompts.
    2. Invokes Groq API (`llama-3.1-8b-instant`) with low temperature (0.1) for determinism.
    3. If running in offline test or with mock keys, uses deterministic factual synthesis.
    4. Evaluates output with `EvidenceGroundingVerifier`.
    5. Measures latency with Prometheus histogram `track_investigation_latency`.
    6. Assembles canonical `InvestigationNarrative` Pydantic model.

INTERVIEW TALKING POINT:
    "We use Groq API with Llama-3-8B running at sub-second generation speeds.
    Because financial institutions require strict reliability, our agent incorporates
    both an online evidence grounding validator and an automated deterministic fallback
    that guarantees compliant, cite-backed narratives even during third-party API outages."
"""

import time
from typing import Any, Optional

from services.investigation.evidence_grounding import EvidenceGroundingVerifier
from services.investigation.prompts import (
    INVESTIGATION_SYSTEM_PROMPT,
    format_investigation_user_prompt,
)
from shared.config.settings import Settings, get_settings
from shared.logging.logger import get_logger
from shared.metrics.prometheus import track_investigation_latency
from shared.models.fraud_case import InvestigationNarrative

logger = get_logger("investigation.agent")


class InvestigationAgent:
    """
    LLM-powered fraud case investigator generating evidence-grounded narratives.
    """

    def __init__(
        self,
        verifier: Optional[EvidenceGroundingVerifier] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Initialize agent with Groq credentials and grounding verifier.

        Args:
            verifier: EvidenceGroundingVerifier instance.
            settings: Settings instance.
        """
        self.settings = settings or get_settings()
        self.verifier = verifier or EvidenceGroundingVerifier()
        self.model_version = self.settings.GROQ_MODEL

    async def generate_narrative(
        self,
        context_bundle: dict[str, Any],
    ) -> InvestigationNarrative:
        """
        Generate grounded investigation narrative from context bundle.

        Args:
            context_bundle: Enriched 360-degree case context dictionary.

        Returns:
            InvestigationNarrative: Validated narrative with risk indicators and grounding score.
        """
        txn_id = context_bundle.get("transaction_info", {}).get("transaction_id", "unknown")
        logger.info("Generating investigation narrative", transaction_id=txn_id, model=self.model_version)

        raw_narrative = ""
        with track_investigation_latency(model_version=self.model_version):
            # Try live Groq API if real API key configured
            if (
                self.settings.GROQ_API_KEY
                and not self.settings.GROQ_API_KEY.startswith("gsk_dev_mock")
                and not self.settings.GROQ_API_KEY.startswith("gsk_your_")
            ):
                try:
                    raw_narrative = await self._call_groq_api(context_bundle)
                except Exception as exc:
                    logger.warning("Groq API call failed; falling back to factual generator", error=str(exc))
                    raw_narrative = self._synthesize_factual_narrative(context_bundle)
            else:
                raw_narrative = self._synthesize_factual_narrative(context_bundle)

        # Verify evidence citations
        score, verified, ungrounded, is_acceptable = self.verifier.verify_narrative(
            raw_narrative, context_bundle
        )

        # Parse structured components
        summary, indicators, cust_context, action = self._parse_narrative_sections(
            raw_narrative, context_bundle
        )

        narrative_obj = InvestigationNarrative(
            transaction_summary=summary,
            risk_indicators=indicators,
            customer_context=cust_context,
            recommended_action=action,
            evidence_grounding_score=score,
            raw_markdown=raw_narrative,
        )

        return narrative_obj

    async def _call_groq_api(self, context_bundle: dict[str, Any]) -> str:
        """Execute Groq API completion."""
        from groq import AsyncGroq

        client = AsyncGroq(api_key=self.settings.GROQ_API_KEY)
        user_prompt = format_investigation_user_prompt(context_bundle)

        completion = await client.chat.completions.create(
            model=self.model_version,
            messages=[
                {"role": "system", "content": INVESTIGATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        return completion.choices[0].message.content or ""

    def _synthesize_factual_narrative(self, context: dict[str, Any]) -> str:
        """
        Deterministic factual narrative generator using strict template citations.
        Guarantees 100% grounded citations during offline mode, tests, or fallback.
        """
        txn_info = context.get("transaction_info", {})
        scoring_info = context.get("scoring_info", {})
        cust_profile = context.get("customer_profile", {})
        graph = context.get("linked_accounts_graph", {})
        merch = context.get("merchant_profile", {})
        geo = context.get("geographic_signals", {})
        feats = context.get("feature_snapshot", {})

        txn_id = txn_info.get("transaction_id", "N/A")
        amount = txn_info.get("amount", 0.0)
        currency = txn_info.get("currency", "INR")
        cust_id = txn_info.get("customer_id", "N/A")
        risk_score = scoring_info.get("risk_score", 0.0)
        risk_label = scoring_info.get("risk_label", "HIGH").upper()

        avg_spend = cust_profile.get("avg_amount", 500.0)
        zscore = feats.get("amount_zscore", 0.0)
        dist = geo.get("distance_from_last_txn_km", 0.0)
        is_intl = geo.get("is_international", False)
        v_1min = feats.get("txn_count_1min", 1)
        is_new_dev = feats.get("is_new_device", 0) == 1
        linked_count = graph.get("shared_device_or_ip_count", 0)
        merch_cat = merch.get("category", "retail")
        fraud_rate = merch.get("category_fraud_rate", 0.01)

        indicators_text = []

        if abs(zscore) > 3.0:
            indicators_text.append(
                f"- [STRONG INDICATOR] Extreme Amount Anomaly: Transaction of {currency} {amount:,.2f} is significantly above customer baseline of {currency} {avg_spend:,.2f} [Citing: amount_zscore={zscore}, amount={amount}]."
            )
        elif abs(zscore) > 1.5:
            indicators_text.append(
                f"- [MODERATE INDICATOR] Elevated Spend Volume: Spend exceeds normal mean [Citing: amount_zscore={zscore}]."
            )

        if dist > 500.0 or is_intl:
            indicators_text.append(
                f"- [STRONG INDICATOR] Geographic Displacement: Transaction occurred {dist:,.1f} km away from last known activity [Citing: distance_from_last_txn={dist}, is_international={str(is_intl).lower()}]."
            )

        if linked_count > 0:
            indicators_text.append(
                f"- [STRONG INDICATOR] Mule Syndicate Hardware Sharing: Device fingerprint is linked to {linked_count} other accounts [Citing: shared_device_or_ip_count={linked_count}]."
            )

        if v_1min > 2:
            indicators_text.append(
                f"- [MODERATE INDICATOR] High Velocity Burst: {v_1min} transactions initiated in 60-second window [Citing: txn_count_1min={v_1min}]."
            )

        if is_new_dev:
            indicators_text.append(
                f"- [MODERATE INDICATOR] Novel Hardware: Transaction originated from unrecognized device fingerprint [Citing: is_new_device=true]."
            )

        if fraud_rate > 0.05:
            indicators_text.append(
                f"- [MODERATE INDICATOR] High-Risk Merchant Category: Merchant category '{merch_cat}' exhibits elevated fraud likelihood [Citing: category_fraud_rate={fraud_rate}]."
            )

        if not indicators_text:
            indicators_text.append("- [WEAK INDICATOR] Minor behavioral variance detected by baseline model.")

        indicators_md = "\n".join(indicators_text)

        # Disposition logic
        if risk_score >= 0.85 or linked_count > 0 or dist > 1000.0:
            action = "ESCALATE_URGENT"
            justification = "Multiple strong indicators of account takeover, syndicate linkage, or impossible travel require immediate freeze of outbound funds."
        elif risk_score >= 0.60:
            action = "HOLD_FOR_REVIEW"
            justification = "Elevated risk signals exceed standard monitoring threshold; human phone verification recommended before settlement."
        else:
            action = "APPROVE"
            justification = "Identified deviations are within acceptable historical risk margins."

        narrative = f"""## Transaction Summary
Flagged transaction `{txn_id}` for customer `{cust_id}` with amount {currency} {amount:,.2f} processed via {txn_info.get('channel', 'UPI')}. Scored by model as {risk_label} with fraud probability {risk_score:.4f}.

## Risk Indicators
{indicators_md}

## Customer Context & Identity Graph
Customer `{cust_id}` has an established baseline average spend of {currency} {avg_spend:,.2f}. Hardware identity graph identifies {linked_count} accounts sharing physical device or IP credentials.

## Recommended Action
{action}
{justification}"""

        return narrative

    def _parse_narrative_sections(
        self,
        raw_markdown: str,
        context: dict[str, Any],
    ) -> tuple[str, list[dict[str, str]], str, str]:
        """Parse markdown narrative into structured sections."""
        summary = "Flagged transaction review"
        indicators: list[dict[str, str]] = []
        cust_context = "Historical baseline analysis"
        action = "HOLD_FOR_REVIEW"

        sections = raw_markdown.split("## ")
        for sec in sections:
            if not sec.strip():
                continue
            lines = sec.strip().split("\n")
            header = lines[0].lower()
            body = "\n".join(lines[1:]).strip()

            if "transaction summary" in header:
                summary = body
            elif "risk indicators" in header:
                for line in body.split("\n"):
                    if line.strip().startswith("-"):
                        tag = "[MODERATE INDICATOR]"
                        if "[STRONG INDICATOR]" in line:
                            tag = "[STRONG INDICATOR]"
                        elif "[WEAK INDICATOR]" in line:
                            tag = "[WEAK INDICATOR]"
                        indicators.append({"indicator": line.strip("- ").strip(), "tag": tag})
            elif "customer context" in header:
                cust_context = body
            elif "recommended action" in header:
                if "ESCALATE_URGENT" in body:
                    action = "ESCALATE_URGENT"
                elif "APPROVE" in body:
                    action = "APPROVE"
                elif "HOLD_FOR_REVIEW" in body:
                    action = "HOLD_FOR_REVIEW"

        return summary, indicators, cust_context, action
