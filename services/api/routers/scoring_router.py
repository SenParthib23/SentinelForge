"""
SentinelForge — services/api/routers/scoring_router.py

PURPOSE:
    High-throughput synchronous real-time fraud scoring endpoint (<150ms SLA).
    Bridges payment authorization requests to the sub-10ms LightGBM engine.

ARCHITECTURE POSITION:
    Layer 5 (FastAPI Gateway) → Real-Time Payment Authorization Path.
"""

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, status

from services.api.dependencies import (
    get_case_builder,
    get_current_user,
    get_fraud_scorer,
    get_investigation_agent,
)
from services.dashboard.sse_streamer import CaseStreamer, get_case_streamer
from services.investigation.case_builder import CaseBuilder
from services.investigation.investigation_agent import InvestigationAgent
from services.scoring.scorer import FraudScorer
from shared.logging.logger import get_logger
from shared.models.fraud_case import FraudScore
from shared.models.transaction import TransactionEvent

logger = get_logger("api.scoring")

scoring_router = APIRouter(prefix="/api/v1/score", tags=["Real-Time Fraud Scoring"])


@scoring_router.post(
    "",
    response_model=FraudScore,
    status_code=status.HTTP_200_OK,
    summary="Synchronous sub-10ms transaction fraud scoring",
)
async def score_transaction_endpoint(
    event: TransactionEvent,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    scorer: Annotated[FraudScorer, Depends(get_fraud_scorer)],
    case_builder: Annotated[CaseBuilder, Depends(get_case_builder)],
    investigation_agent: Annotated[InvestigationAgent, Depends(get_investigation_agent)],
    streamer: Annotated[CaseStreamer, Depends(get_case_streamer)],
) -> FraudScore:
    """
    Score incoming financial transaction synchronously under strict SLA.

    Args:
        event: Validated TransactionEvent payload.
        current_user: Authenticated client claims.
        scorer: Synchronous LightGBM fraud scorer.
        case_builder: Case builder for async context enrichment.
        investigation_agent: LLM agent for background narrative drafting.
        streamer: SSE manager for broadcasting live case updates.

    Returns:
        FraudScore: Prediction probability, operational risk tier, and latencies.
    """
    logger.info(
        "Scoring transaction event",
        transaction_id=event.transaction_id,
        amount=float(event.amount),
        customer_id=event.customer_id,
    )

    # 1. Execute sub-10ms LightGBM scoring
    fraud_score = await scorer.score_transaction(
        transaction_id=event.transaction_id,
        event=event,
    )

    # 2. If flagged (HIGH or CRITICAL), initiate async investigation off the critical path
    if fraud_score.was_flagged:
        asyncio.create_task(
            _run_async_investigation(
                event=event,
                fraud_score=fraud_score,
                case_builder=case_builder,
                investigation_agent=investigation_agent,
                streamer=streamer,
            )
        )

    return fraud_score


async def _run_async_investigation(
    event: TransactionEvent,
    fraud_score: FraudScore,
    case_builder: CaseBuilder,
    investigation_agent: InvestigationAgent,
    streamer: CaseStreamer,
) -> None:
    """
    Asynchronous task: enrich case context, generate LLM narrative, and broadcast SSE.
    """
    txn_id = event.transaction_id
    try:
        # Step A: Build 360-degree context dossier
        context = await case_builder.build_case_context(
            transaction_id=txn_id,
            risk_score=fraud_score.risk_score,
            risk_label=fraud_score.risk_label,
            event=event,
        )

        # Step B: Create case record in PostgreSQL / cache
        case = await case_builder.create_and_persist_case(
            transaction_id=txn_id,
            risk_score=fraud_score.risk_score,
            context_bundle=context,
        )

        # Step C: Broadcast initial 'case_flagged' event to dashboard subscribers
        await streamer.broadcast_event(
            event_type="case_flagged",
            payload={
                "case_id": str(case.id),
                "transaction_id": txn_id,
                "risk_score": fraud_score.risk_score,
                "risk_label": fraud_score.risk_label.value,
                "status": "pending",
            },
            case_id=str(case.id),
        )

        # Step D: Call LLM agent for grounded narrative generation
        narrative = await investigation_agent.generate_narrative(context)

        # Step E: Update case with verified narrative
        await case_builder.update_case_narrative(
            transaction_id=txn_id,
            narrative=narrative.raw_markdown or narrative.transaction_summary,
            narrative_model_version=investigation_agent.model_version,
        )

        # Step F: Broadcast completed narrative to analyst terminals
        await streamer.broadcast_event(
            event_type="narrative_ready",
            payload={
                "case_id": str(case.id),
                "transaction_id": txn_id,
                "grounding_score": narrative.evidence_grounding_score,
                "recommended_action": narrative.recommended_action,
                "narrative": narrative.raw_markdown,
            },
            case_id=str(case.id),
        )
    except Exception as exc:
        logger.error("Async investigation pipeline failed", transaction_id=txn_id, error=str(exc))
