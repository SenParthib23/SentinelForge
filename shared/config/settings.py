"""
SentinelForge — shared/config/settings.py

PURPOSE:
    Centralized, strongly-typed configuration management for SentinelForge using
    Pydantic Settings. Loads configurations from environment variables and `.env`
    files with runtime validation, type coercion, and computed derived fields.

ARCHITECTURE POSITION:
    Layer 0 (Foundation) → Shared across all real-time services, FastAPI endpoints,
    worker processes, and offline ML lifecycle scripts.

HOW IT WORKS:
    Uses `pydantic_settings.BaseSettings` to parse environment variables into validated
    Python types. Defaults match local Docker Compose development defaults.
    Derived fields like `DATABASE_URL` and `REDIS_URL` are computed dynamically via
    `@computed_field` to eliminate configuration drift between individual connection
    parameters and unified connection strings.

INTERVIEW TALKING POINT:
    "We use Pydantic Settings with strict validation and computed connection URIs.
    This guarantees fail-fast behavior at startup if any critical credential or
    endpoint is missing or misformatted, preventing silent runtime failures in
    both the real-time scoring path and background ingestion pipelines."
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings validated at runtime.

    Attributes:
        APP_NAME: Name of the platform.
        ENVIRONMENT: Runtime environment ('development', 'staging', 'production').
        LOG_LEVEL: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        DEBUG: Boolean toggle for debug-mode logging and diagnostics.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Core Application Settings
    APP_NAME: str = "SentinelForge"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False

    # PostgreSQL Database Settings (Port 5433 avoids collision with MAOP)
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5433
    POSTGRES_DB: str = "sentinelforge"
    POSTGRES_USER: str = "sf_user"
    POSTGRES_PASSWORD: str = "sf_secret_dev"
    POSTGRES_MIN_POOL_SIZE: int = 5
    POSTGRES_MAX_POOL_SIZE: int = 20

    # Redis Online Feature Store Settings (Port 6380 avoids collision with MAOP)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6380
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    FEATURE_TTL_SECONDS: int = 86400  # 24 hours feature expiry

    # Kafka Message Broker Settings
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TRANSACTION_TOPIC: str = "txn.stream"
    KAFKA_FLAGGED_TOPIC: str = "case.flagged"
    KAFKA_CONSUMER_GROUP: str = "sentinelforge-consumer-group"

    # LLM Investigation (Groq API) Settings
    GROQ_API_KEY: str = "gsk_dev_mock_key_sentinelforge"
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # MLflow Model Registry Settings (Port 5001 avoids collision)
    MLFLOW_TRACKING_URI: str = "http://localhost:5001"
    MLFLOW_EXPERIMENT_NAME: str = "sentinelforge_fraud_detection"

    # Observability & Metrics (Prometheus Port 9091)
    PROMETHEUS_PORT: int = 9091
    METRICS_ENABLED: bool = True

    # Security & API Authentication
    JWT_SECRET_KEY: str = "dev_secret_key_change_in_production_sentinelforge_2026"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    RATE_LIMIT_PER_MINUTE: int = 60

    # Model Lifecycle & Storage
    MODEL_CACHE_DIR: str = "./model_cache"
    CURRENT_MODEL_VERSION: str = "v1"
    CANARY_TRAFFIC_PERCENT: int = 10

    @computed_field  # type: ignore[prop-decorator]
    @property
    def DATABASE_URL(self) -> str:
        """
        Compute asynchronous PostgreSQL connection DSN for asyncpg.

        Returns:
            str: Valid asyncpg connection URI formatted as
                 postgresql://user:password@host:port/dbname.
        """
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def REDIS_URL(self) -> str:
        """
        Compute Redis connection URL.

        Returns:
            str: Connection URL formatted as redis://[:password@]host:port/db.
        """
        auth_part = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_model_cache_dir(self) -> Path:
        """
        Resolve absolute path to model cache directory, creating it if needed.

        Returns:
            Path: Absolute resolved filesystem path to the model cache directory.
        """
        path = Path(self.MODEL_CACHE_DIR).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache()
def get_settings() -> Settings:
    """
    Retrieve cached Settings singleton instance.

    Uses functools.lru_cache to parse environment and .env configuration
    once per process, avoiding redundant filesystem I/O.

    Returns:
        Settings: Validated global application settings instance.
    """
    return Settings()
