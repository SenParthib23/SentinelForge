"""
SentinelForge — Investigation Package

PURPOSE:
    Provides asynchronous, evidence-grounded fraud investigation assist using
    Groq LLM (Llama-3-8B), 360-degree case context assembly, and automated
    citation grounding verification.
"""

from services.investigation.case_builder import CaseBuilder
from services.investigation.evidence_grounding import EvidenceGroundingVerifier
from services.investigation.investigation_agent import InvestigationAgent
from services.investigation.prompts import (
    INVESTIGATION_SYSTEM_PROMPT,
    format_investigation_user_prompt,
)

__all__ = [
    "CaseBuilder",
    "EvidenceGroundingVerifier",
    "InvestigationAgent",
    "INVESTIGATION_SYSTEM_PROMPT",
    "format_investigation_user_prompt",
]
