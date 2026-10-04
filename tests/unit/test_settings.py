"""
SentinelForge — tests/unit/test_settings.py

PURPOSE:
    Unit tests verifying Pydantic Settings configuration, computed fields,
    environment loading, and port isolation rules.
"""

from shared.config.settings import Settings, get_settings


def test_settings_default_values(mock_settings: Settings) -> None:
    """Verify mock settings initialize with expected values and types."""
    assert mock_settings.ENVIRONMENT == "test"
    assert mock_settings.DEBUG is True
    assert mock_settings.POSTGRES_PORT == 5433
    assert mock_settings.REDIS_PORT == 6380


def test_computed_database_url() -> None:
    """Verify DATABASE_URL correctly formats asyncpg connection DSN."""
    settings = Settings(
        POSTGRES_USER="sf_user",
        POSTGRES_PASSWORD="sf_password",
        POSTGRES_HOST="127.0.0.1",
        POSTGRES_PORT=5433,
        POSTGRES_DB="sentinelforge",
    )
    expected = "postgresql://sf_user:sf_password@127.0.0.1:5433/sentinelforge"
    assert settings.DATABASE_URL == expected


def test_computed_redis_url_without_password() -> None:
    """Verify REDIS_URL generates valid connection URL without credentials."""
    settings = Settings(
        REDIS_HOST="127.0.0.1",
        REDIS_PORT=6380,
        REDIS_DB=0,
        REDIS_PASSWORD=None,
    )
    assert settings.REDIS_URL == "redis://127.0.0.1:6380/0"


def test_computed_redis_url_with_password() -> None:
    """Verify REDIS_URL generates valid connection URL when password is supplied."""
    settings = Settings(
        REDIS_HOST="127.0.0.1",
        REDIS_PORT=6380,
        REDIS_DB=2,
        REDIS_PASSWORD="secret_auth_token",
    )
    assert settings.REDIS_URL == "redis://:secret_auth_token@127.0.0.1:6380/2"


def test_port_isolation_against_maop() -> None:
    """
    Architectural regression test:
    Verify SentinelForge avoids standard ports (5432, 6379, 5000, 9090)
    to allow side-by-side execution with MAOP.
    """
    settings = Settings()
    assert settings.POSTGRES_PORT == 5433, "PostgreSQL must use port 5433 (not 5432)"
    assert settings.REDIS_PORT == 6380, "Redis must use port 6380 (not 6379)"
    assert settings.PROMETHEUS_PORT == 9091, "Prometheus must use port 9091 (not 9090)"
    assert "5001" in settings.MLFLOW_TRACKING_URI, "MLflow must target port 5001 (not 5000)"


def test_get_settings_lru_cache() -> None:
    """Verify get_settings returns identical singleton across invocations."""
    instance_a = get_settings()
    instance_b = get_settings()
    assert instance_a is instance_b
