"""
SentinelForge — services/api/middleware/rate_limiter.py

PURPOSE:
    Redis token-bucket rate limiter middleware safeguarding SentinelForge API
    endpoints against denial-of-service and brute-force attacks.

ARCHITECTURE POSITION:
    Layer 5 (FastAPI Gateway) → Middleware perimeter filtering traffic before routers.

ALGORITHM:
    Sliding window rate limit in Redis: `ratelimit:{client_ip}` with 60-second TTL.
    Exceeding `settings.RATE_LIMIT_PER_MINUTE` returns HTTP 429 Too Many Requests.
"""

import time
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from shared.config.settings import Settings, get_settings
from shared.db.redis_client import RedisClient, get_redis_client
from shared.logging.logger import get_logger

logger = get_logger("api.rate_limiter")


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiting middleware built on Redis or memory fallback.
    """

    def __init__(
        self,
        app,
        redis_client: Optional[RedisClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize middleware."""
        super().__init__(app)
        self.settings = settings or get_settings()
        self.redis = redis_client or get_redis_client()
        # In-memory backup counter: ip -> list of epoch timestamps
        self._local_counts: dict[str, list[float]] = {}

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Check rate limit before dispatching to downstream router.

        Args:
            request: Incoming HTTP request.
            call_next: Downstream handler.

        Returns:
            Response: Standard response or HTTP 429 JSONResponse.
        """
        # Bypass rate limiter for health checks, metrics, and documentation
        path = request.url.path
        if (
            path.startswith("/api/v1/health")
            or path.startswith("/docs")
            or path.startswith("/openapi.json")
            or path.startswith("/redoc")
        ):
            return await call_next(request)

        # Extract client identifier (header or IP)
        client_ip = request.headers.get("X-Forwarded-For")
        if not client_ip and request.client:
            client_ip = request.client.host
        client_key = f"ratelimit:{client_ip or 'unknown'}"

        now = time.time()
        max_limit = self.settings.RATE_LIMIT_PER_MINUTE

        is_allowed = True
        try:
            if self.redis.client is not None or await self.redis.ping():
                # Sliding window via Redis sorted set
                await self.redis.zadd(client_key, {str(now): now})
                await self.redis.expire(client_key, 60)
                # Count in last 60 seconds
                req_count = await self.redis.zcount(client_key, now - 60.0, now)
                # Prune older than 60s
                await self.redis.zremrangebyscore(client_key, 0.0, now - 60.0)

                if req_count > max_limit:
                    is_allowed = False
        except Exception:
            # Memory fallback
            if client_key not in self._local_counts:
                self._local_counts[client_key] = []
            self._local_counts[client_key].append(now)
            self._local_counts[client_key] = [
                t for t in self._local_counts[client_key] if t >= now - 60.0
            ]
            if len(self._local_counts[client_key]) > max_limit:
                is_allowed = False

        if not is_allowed:
            logger.warning("Rate limit exceeded for client", client_ip=client_ip, limit=max_limit)
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "detail": f"Rate limit of {max_limit} requests per minute exceeded.",
                    "retry_after_seconds": 60,
                },
                headers={"Retry-After": "60"},
            )

        return await call_next(request)
