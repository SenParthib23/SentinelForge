"""
SentinelForge — shared/logging/logger.py

PURPOSE:
    Provides production-grade structured logging for all SentinelForge services
    using `structlog`. Emits human-readable colored logs in development and
    machine-readable JSON in production, with contextual trace/request ID injection.

ARCHITECTURE POSITION:
    Layer 0 (Foundation) → Utilized by all streaming workers, API routers,
    scoring pipelines, and ML training/eval scripts for auditability and observability.

HOW IT WORKS:
    Uses Python's contextvars to propagate `request_id` and `transaction_id` across
    asynchronous task boundaries. Configures structlog processors to attach timestamps,
    log levels, caller locations, and execution metrics to every emitted log entry.

INTERVIEW TALKING POINT:
    "In high-throughput fraud scoring systems (10k+ txn/sec), plain text logs
    create debugging nightmares and performance bottlenecks. By using structlog
    with async contextvars, we attach the transaction_id and scoring latency
    to every single log line in structured JSON. This enables instant querying in
    Datadog/Elasticsearch to reconstruct the exact authorization decision path."
"""

import contextvars
import logging
import sys
from typing import Any, Optional

import structlog

# Context variable for thread-safe/coroutine-safe request ID tracking
_request_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)
_transaction_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "transaction_id", default=None
)


def set_request_id(request_id: str) -> None:
    """
    Store request ID in the current coroutine context.

    Args:
        request_id: Unique correlation identifier for HTTP request or event.
    """
    _request_id_ctx.set(request_id)


def set_transaction_id(transaction_id: str) -> None:
    """
    Store transaction ID in the current coroutine context.

    Args:
        transaction_id: Unique financial transaction identifier.
    """
    _transaction_id_ctx.set(transaction_id)


def add_context_identifiers(
    logger: logging.Logger, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """
    Structlog processor that injects current request and transaction IDs.

    Args:
        logger: Underlying standard library logger.
        method_name: Logging method called ('info', 'error', etc.).
        event_dict: Current dictionary of log attributes.

    Returns:
        dict[str, Any]: Augmented dictionary containing active correlation IDs.
    """
    req_id = _request_id_ctx.get()
    if req_id is not None:
        event_dict["request_id"] = req_id

    txn_id = _transaction_id_ctx.get()
    if txn_id is not None:
        event_dict["transaction_id"] = txn_id

    return event_dict


def configure_logging(environment: str = "development", log_level: str = "INFO") -> None:
    """
    Configure global structlog and standard library logging behavior.

    Args:
        environment: 'development' for colorized console output;
                     'production' for compact single-line JSON.
        log_level: Minimum severity level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        add_context_identifiers,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if environment.lower() == "production":
        formatter_processor: structlog.types.Processor = (
            structlog.processors.JSONRenderer()
        )
    else:
        formatter_processor = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            formatter_processor,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Retrieve a bound structlog logger instance with optional module name.

    Args:
        name: Name of the calling module or component.

    Returns:
        structlog.stdlib.BoundLogger: Bound logger ready for structured calls.
    """
    return structlog.get_logger(name)
