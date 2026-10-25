"""
SentinelForge — services/api/routers/investigation_router.py

PURPOSE:
    Investigation case management and human-in-the-loop analyst feedback endpoints.
    Provides case dossiers, SSE event streaming, and feeds analyst adjudications
    directly into the labeled retraining dataset store.

ARCHITECTURE POSITION:
    Layer 5 (FastAPI Gateway) & Layer 6 (Dashboard) → Closes the human feedback flywheel.
"""

import json
from datetime import datetime, timezone
from typing import Annotated, Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sse_starlette.sse import EventSourceResponse

from services.api.dependencies import (
    get_case_builder,
    get_current_user,
)
from services.dashboard.sse_streamer import CaseStreamer, get_case_streamer
from services.investigation.case_builder import CaseBuilder
from shared.db.postgres import PostgresClient, get_postgres_client
from shared.logging.logger import get_logger
from shared.models.fraud_case import (
    AnalystDecision,
    AnalystFeedback,
    CaseStatus,
    FraudCase,
)

logger = get_logger("api.investigation")

investigation_router = APIRouter(prefix="/api/v1/cases", tags=["Fraud Investigation"])


@investigation_router.get(
    "",
    response_model=list[FraudCase],
    summary="List flagged fraud cases for analyst adjudication",
)
async def list_cases(
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    case_builder: Annotated[CaseBuilder, Depends(get_case_builder)],
    postgres: Annotated[PostgresClient, Depends(get_postgres_client)],
    case_status: Optional[CaseStatus] = Query(None, alias="status", description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Page limit"),
) -> list[FraudCase]:
    """
    Retrieve flagged cases awaiting analyst review from database or cache.
    """
    # Try PostgreSQL first if available
    try:
        if postgres.pool is not None or await postgres.is_healthy():
            status_clause = "WHERE status = $1" if case_status else ""
            query = f"""
            SELECT id, transaction_id, risk_score, case_context,
                   investigation_narrative, narrative_model_version,
                   analyst_decision, analyst_notes, analyst_id,
                   decision_timestamp, status, created_at
            FROM fraud_cases
            {status_clause}
            ORDER BY created_at DESC
            LIMIT {limit};
            """
            params = [case_status.value] if case_status else []
            records = await postgres.fetch(query, *params)
            cases: list[FraudCase] = []
            for r in records:
                cases.append(
                    FraudCase(
                        id=r["id"],
                        transaction_id=r["transaction_id"],
                        risk_score=r["risk_score"],
                        case_context=r["case_context"] or {},
                        investigation_narrative=r["investigation_narrative"],
                        narrative_model_version=r["narrative_model_version"],
                        analyst_decision=r["analyst_decision"],
                        analyst_notes=r["analyst_notes"],
                        analyst_id=r["analyst_id"],
                        decision_timestamp=r["decision_timestamp"],
                        status=r["status"],
                        created_at=r["created_at"],
                    )
                )
            return cases
    except Exception as exc:
        logger.debug("Database query fallback to memory cache", error=str(exc))

    # In-memory fallback
    cached = list(case_builder._cases_cache.values())
    if case_status:
        cached = [c for c in cached if c.status == case_status]
    return cached[:limit]


@investigation_router.get(
    "/stream/live",
    summary="Global Server-Sent Events (SSE) live alert feed for all flagged transactions",
)
async def stream_global_events(
    streamer: Annotated[CaseStreamer, Depends(get_case_streamer)],
) -> EventSourceResponse:
    """Subscribe to global real-time event feed for all new fraud alerts."""
    return EventSourceResponse(streamer.subscribe(case_id=None))


@investigation_router.get(
    "/{case_id}",
    response_model=FraudCase,
    summary="Retrieve single case dossier by UUID or transaction ID",
)
async def get_case_detail(
    case_id: str,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    case_builder: Annotated[CaseBuilder, Depends(get_case_builder)],
    postgres: Annotated[PostgresClient, Depends(get_postgres_client)],
) -> FraudCase:
    """Retrieve full case details including context bundle and narrative."""
    # Check cache first
    for c in case_builder._cases_cache.values():
        if str(c.id) == case_id or c.transaction_id == case_id:
            return c

    # Query PostgreSQL
    try:
        if postgres.pool is not None or await postgres.is_healthy():
            query = """
            SELECT id, transaction_id, risk_score, case_context,
                   investigation_narrative, narrative_model_version,
                   analyst_decision, analyst_notes, analyst_id,
                   decision_timestamp, status, created_at
            FROM fraud_cases
            WHERE id::text = $1 OR transaction_id = $1;
            """
            r = await postgres.fetchrow(query, case_id)
            if r:
                return FraudCase(
                    id=r["id"],
                    transaction_id=r["transaction_id"],
                    risk_score=r["risk_score"],
                    case_context=r["case_context"] or {},
                    investigation_narrative=r["investigation_narrative"],
                    narrative_model_version=r["narrative_model_version"],
                    analyst_decision=r["analyst_decision"],
                    analyst_notes=r["analyst_notes"],
                    analyst_id=r["analyst_id"],
                    decision_timestamp=r["decision_timestamp"],
                    status=r["status"],
                    created_at=r["created_at"],
                )
    except Exception as exc:
        logger.debug("Database lookup failed", error=str(exc))

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Fraud case '{case_id}' not found",
    )


@investigation_router.post(
    "/{case_id}/feedback",
    response_model=FraudCase,
    summary="Submit human analyst adjudication and record training label",
)
async def submit_case_feedback(
    case_id: str,
    feedback: AnalystFeedback,
    current_user: Annotated[dict[str, Any], Depends(get_current_user)],
    case_builder: Annotated[CaseBuilder, Depends(get_case_builder)],
    postgres: Annotated[PostgresClient, Depends(get_postgres_client)],
    streamer: Annotated[CaseStreamer, Depends(get_case_streamer)],
) -> FraudCase:
    """
    Record analyst adjudication and persist label into labeled_data table.
    """
    case = None
    for c in case_builder._cases_cache.values():
        if str(c.id) == case_id or c.transaction_id == case_id:
            case = c
            break

    now = datetime.now(timezone.utc)
    analyst_label = "fraud" if feedback.decision == AnalystDecision.REJECTED else "legitimate"

    if case:
        case.analyst_decision = feedback.decision
        case.analyst_notes = feedback.notes
        case.analyst_id = feedback.analyst_id
        case.decision_timestamp = now
        case.status = CaseStatus.RESOLVED

    # Persist adjudication and labeled retraining sample to PostgreSQL
    try:
        if postgres.pool is not None or await postgres.is_healthy():
            # Update fraud_cases
            update_case_sql = """
            UPDATE fraud_cases
            SET analyst_decision = $1,
                analyst_notes = $2,
                analyst_id = $3,
                decision_timestamp = $4,
                status = $5
            WHERE id::text = $6 OR transaction_id = $6;
            """
            await postgres.execute(
                update_case_sql,
                feedback.decision.value,
                feedback.notes,
                feedback.analyst_id,
                now,
                CaseStatus.RESOLVED.value,
                case_id,
            )

            # Insert into labeled_data for ML retraining flywheel
            if case:
                insert_labeled_sql = """
                INSERT INTO labeled_data (transaction_id, features, risk_score, analyst_label, label_source)
                VALUES ($1, $2::jsonb, $3, $4, $5);
                """
                features = case.case_context.get("feature_snapshot", {})
                await postgres.execute(
                    insert_labeled_sql,
                    case.transaction_id,
                    json.dumps(features),
                    case.risk_score,
                    analyst_label,
                    "analyst",
                )
                logger.info(
                    "Recorded analyst feedback into labeled retraining store",
                    transaction_id=case.transaction_id,
                    verdict=feedback.decision.value,
                )
    except Exception as exc:
        logger.debug("PostgreSQL labeled data feedback persistence skipped", error=str(exc))

    # Broadcast event via SSE to all active dashboard clients
    await streamer.broadcast_event(
        event_type="case_resolved",
        payload={
            "case_id": case_id,
            "decision": feedback.decision.value,
            "analyst_id": feedback.analyst_id,
            "status": "resolved",
        },
        case_id=case_id,
    )

    if case:
        return case

    return FraudCase(
        id=UUID(case_id) if "-" in case_id else None,
        transaction_id=case_id,
        risk_score=0.85,
        analyst_decision=feedback.decision,
        analyst_notes=feedback.notes,
        analyst_id=feedback.analyst_id,
        decision_timestamp=now,
        status=CaseStatus.RESOLVED,
    )


@investigation_router.get(
    "/{case_id}/stream",
    summary="Server-Sent Events (SSE) live case update feed",
)
async def stream_case_events(
    case_id: str,
    streamer: Annotated[CaseStreamer, Depends(get_case_streamer)],
) -> EventSourceResponse:
    """
    Subscribe to live real-time investigation progress and collaborative reviews.
    """
    return EventSourceResponse(streamer.subscribe(case_id=case_id))
