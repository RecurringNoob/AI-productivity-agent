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


def _add_logger_name(logger, method_name, event_dict):
    """
    Custom processor to inject the logger name into the event dict.

    structlog.stdlib.add_logger_name only works when the underlying logger
    is a stdlib Logger (has a .name attribute). PrintLogger does not have
    .name, so we extract it from the '_logger' key that structlog injects
    when get_logger(__name__) is called, falling back to "unknown".
    """
    record = event_dict.get("_record")
    if record is not None:
        event_dict["logger"] = record.name
    else:
        # get_logger(__name__) stores the name as the first positional arg
        # in the bound logger's context under the key structlog uses internally.
        # The safest cross-version approach: read from the logger object itself.
        name = getattr(logger, "_logger", None)
        if name is None:
            name = getattr(logger, "name", None)
        # PrintLogger wraps a file; its repr contains no name.
        # Fall back to the module path stored in the context if present.
        event_dict["logger"] = event_dict.pop("logger", name or "app")
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog for structured JSON output to stdout.

    Processors applied in order:
    1. add_log_level       — adds "level" key
    2. _add_logger_name    — adds "logger" key (PrintLogger-safe)
    3. TimeStamper         — adds ISO-8601 "timestamp" key
    4. StackInfoRenderer   — renders stack_info if present
    5. format_exc_info     — renders exception info if present
    6. UnicodeDecoder      — ensures all strings are unicode
    7. JSONRenderer        — serialises the entire event dict to JSON
    """
    processors: list = [
        structlog.stdlib.add_log_level,
        _add_logger_name,
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
log = structlog.get_logger(__name__)