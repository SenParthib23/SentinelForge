"""
SentinelForge — services/api/middleware/request_logger.py

PURPOSE:
    FastAPI middleware capturing request correlation IDs, executing structured
    HTTP access logging, and propagating request context to structlog contextvars.

ARCHITECTURE POSITION:
    Layer 5 (FastAPI Gateway) → Envelopes all incoming HTTP requests.
"""

import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from shared.logging.logger import get_logger, set_request_id

logger = get_logger("api.access")


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    """
    Middleware injecting unique X-Request-ID and logging request execution metrics.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Intercept incoming request, attach correlation ID, and record latency.

        Args:
            request: Starlette/FastAPI request object.
            call_next: Downstream endpoint handler.

        Returns:
            Response: Augmented HTTP response.
        """
        # Extract or generate unique request correlation ID
        request_id = request.headers.get("X-Request-ID", f"req_{uuid4().hex[:12]}")
        set_request_id(request_id)

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # Attach header to response for frontend traceability
            response.headers["X-Request-ID"] = request_id

            logger.info(
                "HTTP Request processed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(elapsed_ms, 2),
                client_ip=request.client.host if request.client else "unknown",
            )
            return response
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                "HTTP Request uncaught exception",
                method=request.method,
                path=request.url.path,
                error=str(exc),
                duration_ms=round(elapsed_ms, 2),
            )
            raise
