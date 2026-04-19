"""
src/logger.py — structlog JSON logging configuration.

Call configure_logging() exactly once at application startup (in FastAPI lifespan).
All modules obtain a logger via:

    import structlog
    log = structlog.get_logger(__name__)

Design Document Reference: Section 4 (Component Breakdown), Phase 5 (Observability)
"""
from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog for structured JSON output to stdout.

    Processors applied in order:
    1. add_log_level  — adds "level" key
    2. add_logger_name — adds "logger" key
    3. TimeStamper    — adds ISO-8601 "timestamp" key
    4. StackInfoRenderer — renders stack_info if present
    5. format_exc_info — renders exception info if present
    6. JSONRenderer   — serialises the entire event dict to JSON
    """
    processors: list = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


# Module-level logger — usable before configure_logging() is called.
# Will output unformatted until configure_logging() runs.
log = structlog.get_logger(__name__)
