"""
src/tools/dateparser_util.py — dateparser wrapper with RELATIVE_BASE injection.

Design Document Reference: Section 9.2 (Date Parser Utility), Improvement 1

Key design decisions:
- RELATIVE_BASE=user_now ensures relative strings like "tomorrow" and "at 10"
  are parsed relative to the USER's current time, not the server's system time.
- RETURN_AS_TIMEZONE_AWARE=True ensures all returned datetimes have tzinfo set.
- Returns None for any input that dateparser cannot resolve.
  Callers MUST handle None by setting intent = "NEEDS_CLARIFICATION".
  (Design Doc §5.2, Improvement 1)
"""
from __future__ import annotations

from datetime import datetime

import dateparser
import structlog

log = structlog.get_logger(__name__)


def parse(raw: str | None, user_now: datetime) -> datetime | None:
    """
    Parse a raw date/time string relative to the user's current time.

    Args:
        raw:      Raw time string extracted from user input or email body.
                  Examples: "tomorrow at 2pm", "next Monday", "2026-05-01 14:00"
        user_now: The user's current datetime (with timezone) used as RELATIVE_BASE.
                  Must be timezone-aware for correct PREFER_DATES_FROM="future" behaviour.

    Returns:
        A timezone-aware datetime if parsing succeeds, or None if the string
        cannot be resolved (ambiguous, empty, or unrecognised).

    Callers must handle None:
        if meeting_dt is None:
            classification.intent = "NEEDS_CLARIFICATION"
    """
    if not raw or not raw.strip():
        return None

    result = dateparser.parse(
        raw,
        settings={
            "PREFER_DATES_FROM": "future",          # Bias toward upcoming dates
            "RETURN_AS_TIMEZONE_AWARE": True,        # Always return tz-aware datetime
            "RELATIVE_BASE": user_now,               # Improvement 1 — user local time
        },
    )

    if result is None:
        log.debug("dateparser_unresolvable", raw=raw)

    return result
