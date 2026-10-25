"""
SentinelForge — Dashboard Package

PURPOSE:
    Provides live analyst dashboard streaming utilities including Server-Sent Events (SSE).
"""

from services.dashboard.sse_streamer import CaseStreamer, get_case_streamer

__all__ = ["CaseStreamer", "get_case_streamer"]
