"""
tests/unit/test_dateparser_util.py — Unit tests for src/tools/dateparser_util.py

Exit criteria (Phase 1, Design Doc):
  ✓ parse("tomorrow at 10", user_now=<9am local>) returns a datetime (not None)
  ✓ parse(None, ...) returns None without raising
  ✓ parse("", ...) returns None
  ✓ RELATIVE_BASE is correctly injected (relative results depend on user_now)
  ✓ Returns timezone-aware datetime
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from src.tools.dateparser_util import parse


class TestParseSafeInputs:
    """None and empty-string inputs must return None, never raise."""

    def test_none_input_returns_none(self):
        user_now = datetime(2026, 4, 14, 9, 0, tzinfo=timezone.utc)
        assert parse(None, user_now) is None

    def test_empty_string_returns_none(self):
        user_now = datetime(2026, 4, 14, 9, 0, tzinfo=timezone.utc)
        assert parse("", user_now) is None

    def test_whitespace_string_returns_none(self):
        user_now = datetime(2026, 4, 14, 9, 0, tzinfo=timezone.utc)
        assert parse("   ", user_now) is None


class TestParseRelativeDates:
    """Relative strings must be resolved using user_now as the base."""

    def test_tomorrow_at_10am(self):
        """Core exit criterion: 'tomorrow at 10am' should parse successfully."""
        user_now = datetime(2026, 4, 14, 9, 0, tzinfo=timezone.utc)
        result = parse("tomorrow at 10am", user_now)
        assert result is not None
        assert result.day == 15  # tomorrow relative to April 14
        assert result.hour == 10

    def test_relative_base_injected_april(self):
        """RELATIVE_BASE=April 1 → 'tomorrow' = April 2."""
        user_now_april = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
        result = parse("tomorrow", user_now_april)
        assert result is not None
        assert result.month == 4
        assert result.day == 2

    def test_relative_base_injected_june(self):
        """RELATIVE_BASE=June 1 → 'tomorrow' = June 2."""
        user_now_june = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
        result = parse("tomorrow", user_now_june)
        assert result is not None
        assert result.month == 6
        assert result.day == 2

    def test_user_now_differs_yields_different_results(self):
        """Two different user_now values must yield two different parse results."""
        now_1 = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
        now_2 = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
        result_1 = parse("tomorrow", now_1)
        result_2 = parse("tomorrow", now_2)
        assert result_1 is not None
        assert result_2 is not None
        assert result_1 != result_2


class TestParseAbsoluteDates:
    """Absolute date strings should parse regardless of user_now."""

    def test_iso_datetime_string(self):
        user_now = datetime(2026, 4, 14, 9, 0, tzinfo=timezone.utc)
        result = parse("2026-05-01 14:00", user_now)
        assert result is not None
        assert result.month == 5
        assert result.day == 1
        assert result.hour == 14

    def test_natural_language_date(self):
        user_now = datetime(2026, 4, 14, 9, 0, tzinfo=timezone.utc)
        result = parse("May 1 2026 at 2pm", user_now)
        assert result is not None
        assert result.month == 5


class TestParseReturnType:
    """Successful parses must always return timezone-aware datetimes."""

    def test_returns_timezone_aware(self):
        user_now = datetime(2026, 4, 14, 9, 0, tzinfo=timezone.utc)
        result = parse("tomorrow at 3pm", user_now)
        assert result is not None
        assert result.tzinfo is not None, "Returned datetime must be timezone-aware"

    def test_returns_datetime_type(self):
        user_now = datetime(2026, 4, 14, 9, 0, tzinfo=timezone.utc)
        result = parse("next monday at 9am", user_now)
        assert result is None or isinstance(result, datetime)
