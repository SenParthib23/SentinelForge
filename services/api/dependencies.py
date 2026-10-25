"""
SentinelForge — services/api/dependencies.py

PURPOSE:
    FastAPI dependency injection providers for JWT authentication, database pools,
    scoring engines, and investigation services.

ARCHITECTURE POSITION:
    Layer 5 (FastAPI Gateway) → Provides reusable injected singletons for API routers.
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from services.dashboard.sse_streamer import CaseStreamer, get_case_streamer
from services.investigation.case_builder import CaseBuilder
from services.investigation.investigation_agent import InvestigationAgent
from services.scoring.model_loader import ModelLoader, get_model_loader
from services.scoring.scorer import FraudScorer
from shared.config.settings import Settings, get_settings
from shared.db.postgres import PostgresClient, get_postgres_client
from shared.db.redis_client import RedisClient, get_redis_client

security_scheme = HTTPBearer(auto_error=False)


def create_access_token(
    data: dict[str, Any],
    expires_delta: Optional[timedelta] = None,
    settings: Optional[Settings] = None,
) -> str:
    """
    Generate signed JWT bearer token.

    Args:
        data: Claims payload (e.g. `{"sub": username, "role": role}`).
        expires_delta: Optional token expiration lifetime.
        settings: Settings instance.

    Returns:
        str: Encoded JWT token string.
    """
    cfg = settings or get_settings()
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=cfg.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, cfg.JWT_SECRET_KEY, algorithm=cfg.JWT_ALGORITHM)


async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Security(security_scheme)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict[str, Any]:
    """
    Validate incoming Authorization: Bearer <token> header.

    Args:
        credentials: HTTP Authorization credentials.
        settings: Injected settings.

    Returns:
        dict[str, Any]: Decoded token claims (user identity).

    Raises:
        HTTPException 401: If token is missing, expired, or signature is invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        username: str = payload.get("sub", "")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing subject identifier",
            )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired JWT authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Singleton service instances for FastAPI dependency injection
_fraud_scorer: Optional[FraudScorer] = None
_case_builder: Optional[CaseBuilder] = None
_investigation_agent: Optional[InvestigationAgent] = None


def get_fraud_scorer() -> FraudScorer:
    """Provide singleton FraudScorer instance."""
    global _fraud_scorer
    if _fraud_scorer is None:
        _fraud_scorer = FraudScorer()
    return _fraud_scorer


def get_case_builder() -> CaseBuilder:
    """Provide singleton CaseBuilder instance."""
    global _case_builder
    if _case_builder is None:
        _case_builder = CaseBuilder()
    return _case_builder


def get_investigation_agent() -> InvestigationAgent:
    """Provide singleton InvestigationAgent instance."""
    global _investigation_agent
    if _investigation_agent is None:
        _investigation_agent = InvestigationAgent()
    return _investigation_agent
