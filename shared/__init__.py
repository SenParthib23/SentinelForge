"""
SentinelForge — Shared Package

PURPOSE:
    Provides shared configuration, database clients, data models,
    structured logging, and Prometheus observability metrics across
    all SentinelForge services and offline ML pipelines.

ARCHITECTURE POSITION:
    Foundation layer imported by both streaming services (Layers 1-6)
    and offline ML lifecycle components (Layers 7-10).
"""
