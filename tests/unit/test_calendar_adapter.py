"""
tests/unit/test_calendar_adapter.py — Unit tests for SQLiteCalendarAdapter.

Each test runs against a fresh in-memory SQLite DB (via the test_db autouse fixture).

Exit criteria (Phase 1, Design Doc):
  ✓ check_overlap catches true overlap
  ✓ check_overlap catches exact-time match
  ✓ check_overlap catches buffer-only soft conflict (Improvement 4)
  ✓ Adjacent events with 0-min buffer do NOT conflict
  ✓ safe_schedule is atomic: creates event on free slot, returns ConflictInfo on busy slot
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from src.db import Event
from src.parsers import ConflictInfo
from src.tools.calendar import SQLiteCalendarAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SLOT_START = datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc)
SLOT_END   = datetime(2026, 5, 1, 16, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# check_overlap tests
# ---------------------------------------------------------------------------


class TestCheckOverlap:
    """Tests for the range-based overlap query (Design Doc §9.1)."""

    async def test_returns_none_when_calendar_empty(self, calendar_adapter):
        result = await calendar_adapter.check_overlap(SLOT_START, SLOT_END)
        assert result is None

    async def test_returns_none_for_non_overlapping_event(self, calendar_adapter):
        # Event 2–4 PM; check 5–7 PM → no overlap
        await calendar_adapter.create_event("Existing", SLOT_START, SLOT_END, "manual")
        later_start = SLOT_END + timedelta(hours=1)
        later_end   = later_start + timedelta(hours=2)
        result = await calendar_adapter.check_overlap(later_start, later_end)
        assert result is None

    async def test_detects_exact_overlap(self, calendar_adapter):
        await calendar_adapter.create_event("Existing", SLOT_START, SLOT_END, "manual")
        result = await calendar_adapter.check_overlap(SLOT_START, SLOT_END)
        assert result is not None
        assert isinstance(result, ConflictInfo)
        assert result.conflicting_title == "Existing"

    async def test_detects_partial_overlap_starts_before(self, calendar_adapter):
        await calendar_adapter.create_event("Existing", SLOT_START, SLOT_END, "manual")
        # New event starts 1 hour before and ends in the middle
        new_start = SLOT_START - timedelta(hours=1)
        new_end   = SLOT_START + timedelta(hours=1)
        result = await calendar_adapter.check_overlap(new_start, new_end)
        assert result is not None

    async def test_detects_partial_overlap_starts_after(self, calendar_adapter):
        await calendar_adapter.create_event("Existing", SLOT_START, SLOT_END, "manual")
        # New event starts in the middle and ends after
        new_start = SLOT_START + timedelta(hours=1)
        new_end   = SLOT_END   + timedelta(hours=1)
        result = await calendar_adapter.check_overlap(new_start, new_end)
        assert result is not None

    async def test_detects_contained_event(self, calendar_adapter):
        await calendar_adapter.create_event("Existing", SLOT_START, SLOT_END, "manual")
        # New event is entirely inside existing
        new_start = SLOT_START + timedelta(minutes=30)
        new_end   = SLOT_END   - timedelta(minutes=30)
        result = await calendar_adapter.check_overlap(new_start, new_end)
        assert result is not None

    async def test_no_conflict_for_adjacent_events_zero_buffer(self, calendar_adapter):
        """
        Adjacent events (end of first == start of second) with buffer=0
        must NOT be considered a conflict. (Improvement 4 — buffer=0 baseline)
        """
        await calendar_adapter.create_event("Event 1", SLOT_START, SLOT_END, "manual")
        # New event starts exactly when the first ends
        result = await calendar_adapter.check_overlap(SLOT_END, SLOT_END + timedelta(hours=2), buffer_minutes=0)
        assert result is None

    async def test_buffer_catches_soft_conflict(self, calendar_adapter):
        """
        Improvement 4: event ending at 6 PM + new event at 6:15 PM with a 20-min
        buffer → conflict detected (6:15 PM - 20 min = 5:55 PM < 6 PM).
        """
        event_start = datetime(2026, 5, 1, 16, 0, tzinfo=timezone.utc)
        event_end   = datetime(2026, 5, 1, 18, 0, tzinfo=timezone.utc)
        await calendar_adapter.create_event("Event at 4pm", event_start, event_end, "manual")

        new_start = datetime(2026, 5, 1, 18, 15, tzinfo=timezone.utc)  # 15 min after
        new_end   = datetime(2026, 5, 1, 19,  0, tzinfo=timezone.utc)

        # 0-min buffer: no conflict
        assert await calendar_adapter.check_overlap(new_start, new_end, buffer_minutes=0) is None

        # 20-min buffer: conflict (buffer expands query to 5:55 PM, which overlaps 4-6 PM)
        conflict = await calendar_adapter.check_overlap(new_start, new_end, buffer_minutes=20)
        assert conflict is not None
        assert conflict.buffer_applied == 20

    async def test_conflict_info_has_correct_fields(self, calendar_adapter):
        await calendar_adapter.create_event("My Meeting", SLOT_START, SLOT_END, "meeting")
        conflict = await calendar_adapter.check_overlap(SLOT_START, SLOT_END)
        assert conflict is not None
        assert conflict.conflicting_title == "My Meeting"
        assert conflict.conflicting_start == SLOT_START
        assert conflict.conflicting_end   == SLOT_END
        assert conflict.buffer_applied == 0


# ---------------------------------------------------------------------------
# create_event tests
# ---------------------------------------------------------------------------


class TestCreateEvent:

    async def test_creates_event_with_correct_fields(self, calendar_adapter):
        event = await calendar_adapter.create_event(
            "Contest Round 1", SLOT_START, SLOT_END, "contest", priority=1
        )
        assert event.id is not None
        assert event.title    == "Contest Round 1"
        assert event.source   == "contest"
        assert event.priority == 1

    async def test_creates_event_with_external_id(self, calendar_adapter):
        event = await calendar_adapter.create_event(
            "CF Round 987", SLOT_START, SLOT_END, "contest", external_id="codeforces-987"
        )
        assert event.external_id == "codeforces-987"

    async def test_default_priority_is_zero(self, calendar_adapter):
        event = await calendar_adapter.create_event("Low Prio", SLOT_START, SLOT_END, "manual")
        assert event.priority == 0


# ---------------------------------------------------------------------------
# safe_schedule tests
# ---------------------------------------------------------------------------


class TestSafeSchedule:
    """Atomic check-then-write tests (Design Doc §9.1, Improvement 2)."""

    async def test_schedules_when_free(self, calendar_adapter):
        result = await calendar_adapter.safe_schedule(
            "Contest A", SLOT_START, SLOT_END, "contest"
        )
        assert isinstance(result, Event)
        assert result.title == "Contest A"

    async def test_returns_conflict_when_busy(self, calendar_adapter):
        await calendar_adapter.create_event("Existing", SLOT_START, SLOT_END, "manual")
        result = await calendar_adapter.safe_schedule(
            "New Event", SLOT_START, SLOT_END, "meeting"
        )
        assert isinstance(result, ConflictInfo)
        assert result.conflicting_title == "Existing"

    async def test_does_not_create_event_on_conflict(self, calendar_adapter):
        """After a conflict, only the original event exists in the DB."""
        await calendar_adapter.create_event("Existing", SLOT_START, SLOT_END, "manual")
        await calendar_adapter.safe_schedule("Blocked", SLOT_START, SLOT_END, "meeting")

        # Only one event in DB
        conflict = await calendar_adapter.check_overlap(SLOT_START, SLOT_END)
        assert conflict is not None
        assert conflict.conflicting_title == "Existing"  # not "Blocked"

    async def test_safe_schedule_with_buffer(self, calendar_adapter):
        """safe_schedule passes buffer_minutes to check_overlap."""
        event_start = datetime(2026, 5, 1, 16, 0, tzinfo=timezone.utc)
        event_end   = datetime(2026, 5, 1, 18, 0, tzinfo=timezone.utc)
        await calendar_adapter.create_event("Existing", event_start, event_end, "manual")

        new_start = datetime(2026, 5, 1, 18, 15, tzinfo=timezone.utc)
        new_end   = datetime(2026, 5, 1, 19,  0, tzinfo=timezone.utc)

        # With 20-min buffer → conflict
        result = await calendar_adapter.safe_schedule(
            "Meeting", new_start, new_end, "meeting", buffer_minutes=20
        )
        assert isinstance(result, ConflictInfo)
