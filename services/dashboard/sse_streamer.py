"""
SentinelForge — services/dashboard/sse_streamer.py

PURPOSE:
    Provides real-time event broadcasting to the Fraud Analyst Dashboard via
    Server-Sent Events (SSE). Streams live case status transitions, progressive
    LLM narrative generation, and collaborative review adjudications.

ARCHITECTURE POSITION:
    Layer 6 (Analyst Dashboard / SSE Gateway) → Bridges async backend events
    with frontend analyst terminals.

HOW IT WORKS:
    Maintains an in-memory Pub/Sub fan-out using asyncio.Queue per subscriber.
    When a case transitions (e.g. `case.flagged` -> `context_ready` -> `narrative_ready`),
    subscribers connected to `GET /api/v1/cases/{case_id}/stream` receive live JSON chunks.

INTERVIEW TALKING POINT:
    "We chose Server-Sent Events (SSE) over full-duplex WebSockets because investigation
    assist is unidirectional: the backend pushes case assembly progress and streaming
    LLM tokens to the analyst dashboard. SSE runs natively over HTTP/2, traverses corporate
    firewalls without custom handshake proxies, and supports auto-reconnection natively."
"""

import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

from shared.logging.logger import get_logger

logger = get_logger("dashboard.sse")


class CaseStreamer:
    """
    Pub/Sub event fan-out manager for streaming live case updates over SSE.
    """

    def __init__(self) -> None:
        """Initialize subscriber queue registry."""
        # case_id -> set of asyncio.Queue instances
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)
        # global subscribers receiving every new case alert
        self._global_subscribers: set[asyncio.Queue] = set()

    async def subscribe(self, case_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Subscribe to case-specific or global real-time event feed.

        Args:
            case_id: Specific case ID to listen for, or None for global feed.

        Yields:
            str: SSE formatted event message: `data: {...}\n\n`
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)

        if case_id:
            self._subscribers[case_id].add(queue)
            logger.info("New analyst subscribed to case stream", case_id=case_id)
        else:
            self._global_subscribers.add(queue)
            logger.info("New analyst subscribed to global alerts stream")

        try:
            # Send initial ping event
            init_payload = {
                "event": "connected",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "case_id": case_id or "global",
            }
            yield f"data: {json.dumps(init_payload)}\n\n"

            while True:
                data = await queue.get()
                yield f"data: {json.dumps(data)}\n\n"
        except asyncio.CancelledError:
            logger.debug("Analyst SSE connection closed", case_id=case_id)
        finally:
            if case_id and queue in self._subscribers[case_id]:
                self._subscribers[case_id].remove(queue)
                if not self._subscribers[case_id]:
                    del self._subscribers[case_id]
            elif queue in self._global_subscribers:
                self._global_subscribers.remove(queue)

    async def broadcast_event(
        self,
        event_type: str,
        payload: dict[str, Any],
        case_id: Optional[str] = None,
    ) -> None:
        """
        Broadcast structured event to active subscribers.

        Args:
            event_type: Event category (e.g. 'case_flagged', 'narrative_ready', 'case_resolved').
            payload: Event data dictionary.
            case_id: Optional target case ID.
        """
        msg = {
            "event": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": payload,
        }

        # Deliver to specific case subscribers
        if case_id and case_id in self._subscribers:
            for q in list(self._subscribers[case_id]):
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    pass

        # Deliver to global subscribers
        for q in list(self._global_subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass


_case_streamer: Optional[CaseStreamer] = None


def get_case_streamer() -> CaseStreamer:
    """
    Retrieve global CaseStreamer singleton instance.

    Returns:
        CaseStreamer: Shared singleton.
    """
    global _case_streamer
    if _case_streamer is None:
        _case_streamer = CaseStreamer()
    return _case_streamer
