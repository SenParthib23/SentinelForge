"""
SentinelForge — services/api/main.py

PURPOSE:
    FastAPI gateway entrypoint. Configures application lifespan, database connection
    pooling, CORS policies, rate limiting, request tracing, and router mounts.

ARCHITECTURE POSITION:
    Layer 5 (FastAPI Gateway) → Main external entry point for synchronous scoring
    and analyst dashboard interactions.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from services.api.middleware.rate_limiter import RateLimiterMiddleware
from services.api.middleware.request_logger import RequestLoggerMiddleware
from services.api.routers.auth_router import auth_router
from services.api.routers.health_router import health_router
from services.api.routers.investigation_router import investigation_router
from services.api.routers.model_router import model_router
from services.api.routers.scoring_router import scoring_router
from services.scoring.model_loader import get_model_loader
from shared.config.settings import get_settings
from shared.db.postgres import get_postgres_client
from shared.db.redis_client import get_redis_client
from shared.logging.logger import configure_logging, get_logger

# Configure structured logging
settings = get_settings()
configure_logging(environment=settings.ENVIRONMENT, log_level=settings.LOG_LEVEL)
logger = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage startup and shutdown lifecycle hooks.

    Initializes database pools and preloads LightGBM models into memory;
    safely closes connections on server shutdown.
    """
    logger.info("Starting SentinelForge API Gateway", environment=settings.ENVIRONMENT)

    # 1. Initialize PostgreSQL pool
    postgres = get_postgres_client()
    try:
        await postgres.connect()
    except Exception as exc:
        logger.warning("PostgreSQL connection deferred (standalone mode)", error=str(exc))

    # 2. Initialize Redis client
    redis = get_redis_client()
    try:
        await redis.connect()
    except Exception as exc:
        logger.warning("Redis connection deferred (standalone mode)", error=str(exc))

    # 3. Preload LightGBM model into memory
    model_loader = get_model_loader()
    try:
        model_loader.load_model()
        logger.info("Preloaded scoring model into memory", version=model_loader.current_version)
    except Exception as exc:
        logger.warning("Model preload failed", error=str(exc))

    yield

    # Shutdown hooks
    logger.info("Shutting down SentinelForge API Gateway...")
    await postgres.disconnect()
    await redis.disconnect()
    logger.info("Connections closed cleanly")


def create_app() -> FastAPI:
    """
    Application factory instantiating and configuring FastAPI instance.

    Returns:
        FastAPI: Configured web application instance.
    """
    app = FastAPI(
        title="SentinelForge Fraud Detection API",
        description=(
            "Production-grade, real-time transaction fraud detection system with sub-10ms "
            "LightGBM scoring, streaming behavioral feature engine, and async Groq LLM "
            "investigation assist."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # 1. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Custom Access Logger and Rate Limiter Middlewares
    app.add_middleware(RateLimiterMiddleware)
    app.add_middleware(RequestLoggerMiddleware)

    # 3. Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled server exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred while processing the request.",
                "path": request.url.path,
            },
        )

    # 4. Mount Routers
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(scoring_router)
    app.include_router(investigation_router)
    app.include_router(model_router)

    # 5. Mount Static Dashboard
    dashboard_static = Path(__file__).resolve().parent.parent / "dashboard" / "static"
    if dashboard_static.exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/static", StaticFiles(directory=str(dashboard_static)), name="static")

        @app.get("/", include_in_schema=False)
        @app.get("/dashboard", include_in_schema=False)
        async def serve_dashboard() -> FileResponse:
            return FileResponse(dashboard_static / "index.html")

    return app


app = create_app()
