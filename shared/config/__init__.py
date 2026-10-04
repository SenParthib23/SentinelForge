"""
SentinelForge — Configuration Package

PURPOSE:
    Provides centralized environment-driven configuration management.
"""

from shared.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
