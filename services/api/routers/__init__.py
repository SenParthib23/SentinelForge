"""
SentinelForge — API Routers Package
"""

from services.api.routers.auth_router import auth_router
from services.api.routers.health_router import health_router
from services.api.routers.investigation_router import investigation_router
from services.api.routers.model_router import model_router
from services.api.routers.scoring_router import scoring_router

__all__ = [
    "auth_router",
    "health_router",
    "scoring_router",
    "investigation_router",
    "model_router",
]
