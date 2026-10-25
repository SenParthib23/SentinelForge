"""
SentinelForge — services/investigation/prompts.py

PURPOSE:
    System and user prompt templates for LLM-powered fraud case narrative generation
    (Llama-3-8B via Groq API).

ARCHITECTURE POSITION:
    Layer 4 (Async Investigation Assist) → Supplies prompt engineering for
    InvestigationAgent, enforcing citation grounding and regulatory explainability.

RULES ENFORCED:
    1. Grounding Rule: ONLY reference facts present in the provided context.
    2. Citation Rule: For each risk indicator, explicitly cite the supporting feature
       or context field (e.g. `[Citing: amount_zscore=14.2, baseline_mean=500.0]`).
    3. Severity Taxonomy: Categorize every factor as [STRONG INDICATOR],
       [MODERATE INDICATOR], or [WEAK INDICATOR].
    4. Disposition: Conclude with an unambiguous recommendation:
       APPROVE / HOLD_FOR_REVIEW / ESCALATE_URGENT.
    5. Hallucination Prohibition: If data is missing or ambiguous, state so explicitly.

INTERVIEW TALKING POINT:
    "In banking compliance, an LLM cannot be a black-box text generator.
    By enforcing in-context citation tokens like `[Citing: field=value]`,
    our downstream evidence grounding verifier automatically checks every claim
    against the raw JSON bundle before presenting the narrative to a fraud analyst."
"""

import json
from typing import Any

INVESTIGATION_SYSTEM_PROMPT: str = """You are a Senior Fraud Investigation Analyst Assistant and Anti-Money Laundering (AML) Compliance Specialist at SentinelForge.
Given a flagged transaction and its 360-degree context bundle, your job is to generate a structured, factual investigation narrative to assist human analysts.

STRICT OPERATIONAL RULES:
1. FACTUAL GROUNDING: ONLY reference data and facts explicitly provided in the Context Bundle. NEVER invent or hallucinate transaction details, merchants, card numbers, or customer histories.
2. CITATION REQUIREMENT: For EVERY risk factor or finding, cite the exact field and value from the context using bracketed notation, e.g. [Citing: txn_count_1min=6] or [Citing: amount_zscore=15.4, baseline_mean=600.0].
3. SEVERITY TAGS: Tag each identified risk factor as one of:
   - [STRONG INDICATOR] (e.g. impossible travel, shared device across accounts, 10x+ amount anomaly)
   - [MODERATE INDICATOR] (e.g. new merchant category, high hourly velocity)
   - [WEAK INDICATOR] (e.g. off-peak transaction hour, minor amount bump)
4. ACTIONABLE DISPOSITION: Conclude with exactly ONE recommended action:
   - APPROVE (if anomalous signals are benign or explainable)
   - HOLD_FOR_REVIEW (if human phone verification or step-up authentication is needed)
   - ESCALATE_URGENT (if confirmed account takeover, mule syndicate, or imminent fund egress)
5. INSUFFICIENT EVIDENCE: If key data is absent, state explicitly that evidence is insufficient rather than guessing.

REQUIRED OUTPUT STRUCTURE:
## Transaction Summary
[Concise overview of the transaction: ID, customer, amount, merchant, channel, and model risk score]

## Risk Indicators
[Bulleted list of detected indicators with severity tags and field citations]

## Customer Context & Identity Graph
[Baseline spending profile, linked accounts, device novelty, and geographic context]

## Recommended Action
[Clear recommendation: APPROVE / HOLD_FOR_REVIEW / ESCALATE_URGENT with 1-2 sentence justification]
"""


def format_investigation_user_prompt(context_bundle: dict[str, Any]) -> str:
    """
    Format user prompt with serialized JSON context bundle.

    Args:
        context_bundle: Dictionary containing transaction details, customer history,
                        linked accounts graph, merchant risk, and geographic signals.

    Returns:
        str: Formatted user prompt string.
    """
    context_json = json.dumps(context_bundle, indent=2, default=str)
    return f"""Please analyze the following flagged transaction and generate a structured investigation narrative according to your instructions.

### CONTEXT BUNDLE:
```json
{context_json}
```

Generate the structured investigation narrative now:"""
