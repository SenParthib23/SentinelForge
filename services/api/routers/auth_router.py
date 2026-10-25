"""
SentinelForge — services/api/routers/auth_router.py

PURPOSE:
    Authentication endpoint issuing signed JWT Bearer tokens for fraud analysts
    and automated gateway systems.

ARCHITECTURE POSITION:
    Layer 5 (FastAPI Gateway) → Security gatekeeper.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from services.api.dependencies import create_access_token
from shared.config.settings import Settings, get_settings
from shared.logging.logger import get_logger

logger = get_logger("api.auth")

auth_router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    """Credentials payload for JWT token issuance."""

    username: str = Field(..., description="Analyst or service account username")
    password: str = Field(..., description="Account secret password")


class TokenResponse(BaseModel):
    """Bearer token authentication response."""

    access_token: str = Field(..., description="Signed JWT Bearer token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in_minutes: int = Field(..., description="Validity lifetime in minutes")
    username: str = Field(..., description="Authenticated user identity")
    role: str = Field(..., description="Assigned role (e.g. analyst, admin)")


# Demo and development credential store
DEMO_USERS: dict[str, dict[str, str]] = {
    "admin": {"password": "sentinelforge2026", "role": "admin"},
    "analyst_jane": {"password": "sentinelforge2026", "role": "analyst"},
    "gateway_service": {"password": "sentinelforge2026", "role": "service"},
}


@auth_router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate and receive JWT Bearer token",
)
async def login(
    payload: LoginRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> TokenResponse:
    """
    Authenticate analyst credentials and issue a signed JWT access token.

    Args:
        payload: Username and password.
        settings: Application settings.

    Returns:
        TokenResponse: Access token and metadata.

    Raises:
        HTTPException 401: If username or password is invalid.
    """
    user_info = DEMO_USERS.get(payload.username)

    if not user_info or user_info["password"] != payload.password:
        logger.warning("Failed login attempt", username=payload.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    claims = {
        "sub": payload.username,
        "role": user_info["role"],
    }
    token = create_access_token(claims, settings=settings)

    logger.info("User authenticated successfully", username=payload.username, role=user_info["role"])

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        username=payload.username,
        role=user_info["role"],
    )
