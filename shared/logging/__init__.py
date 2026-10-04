"""
SentinelForge — Logging Package

PURPOSE:
    Provides structured, contextual JSON/console logging using structlog.
"""

from shared.logging.logger import configure_logging, get_logger, set_request_id

__all__ = ["configure_logging", "get_logger", "set_request_id"]
