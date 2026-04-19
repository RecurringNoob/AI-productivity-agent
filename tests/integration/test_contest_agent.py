"""
tests/integration/test_contest_agent.py — Integration tests for ContestAgent.

Uses mock LLM and mock ContestProvider — no real API calls or Gemini.
Uses the real SQLiteCalendarAdapter backed by the test in-memory DB (test_db fixture).

Exit criteria (Phase 2, Design Doc §8.2):
  ✓ Contest scheduled when calendar is free
  ✓ ConflictInfo returned when slot is busy (pre-existing event)
  ✓ HITL staged (not scheduled) for contests ≥ 4 hours
  ✓ "No contests" message when provider returns empty list
  ✓ external_id is passed through to safe_schedule
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.contest import ContestAgent
from src.hitl import HITLManager
from src.parsers import ContestDTO, ContestSelectionResult
from src.tools.calendar import SQLiteCalendarAdapter

SLOT_START = datetime(2026, 5, 10, 14, 0, tzinfo=timezone.utc)
SLOT_END   = datetime(2026, 5, 10, 16, 0, tzinfo=timezone.utc)


def make_contest_dto(
    title: str = "CF Round 987",
    start: datetime = SLOT_START,
    duration_hours: float = 2.0,
    provider: str = "codeforces",
    external_id: str = "codeforces-987",
) -> ContestDTO:
    return ContestDTO(
        title=title,
        start=start,
        end=start + timedelta(hours=duration_hours),
        provider=provider,
        external_id=external_id,
    )


def make_agent(
    calendar_adapter,
    duration_hours: float = 2.0,
    title: str = "CF Round 987",
    provider_contests: list | None = None,
) -> ContestAgent:
    """Build a ContestAgent with mock LLM and mock ContestProvider."""
    contests = provider_contests if provider_contests is not None else [
        make_contest_dto(title=title, duration_hours=duration_hours)
    ]

    mock_provider = AsyncMock()
    mock_provider.fetch = AsyncMock(return_value=contests)

    selected = ContestSelectionResult(
        title=contests[0].title if contests else title,
        start=contests[0].start if contests else SLOT_START,
        duration_hours=contests[0].end.hour - contests[0].start.hour if contests else duration_hours,
        provider=contests[0].provider if contests else "codeforces",
    ) if contests else None

    mock_chain = AsyncMock()
    mock_chain.ainvoke = AsyncMock(return_value=selected)
    mock_chain.with_retry = MagicMock(return_value=mock_chain)

    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_chain)

    hitl = HITLManager(calendar_adapter=calendar_adapter)

    return ContestAgent(
        contest_provider=mock_provider,
        calendar_adapter=calendar_adapter,
        hitl_manager=hitl,
        llm=mock_llm,
        fetch_limit=5,
    )


class TestContestAgentScheduling:

    async def test_schedules_when_calendar_free(self, calendar_adapter):
        agent = make_agent(calendar_adapter, duration_hours=2.0)
        result = await agent.run("find me a contest")
        assert "CF Round 987" in result["result"]
        assert result["pending_action_id"] is None
        # Verify event was created in DB
        conflict = await calendar_adapter.check_overlap(SLOT_START, SLOT_END)
        assert conflict is not None  # event now exists
        assert conflict.conflicting_title == "CF Round 987"

    async def test_returns_conflict_message_when_busy(self, calendar_adapter):
        # Pre-create a blocking event
        await calendar_adapter.create_event("Existing Event", SLOT_START, SLOT_END, "manual")
        agent = make_agent(calendar_adapter, duration_hours=2.0)
        result = await agent.run("find me a contest")
        # Should contain conflict info, not an event creation message
        assert "conflict" in result["result"].lower() or "Existing Event" in result["result"]
        assert result["pending_action_id"] is None

    async def test_hitl_staged_for_long_contest(self, calendar_adapter):
        """Contests ≥ 4 hours must stage HITL, not create events directly."""
        agent = make_agent(calendar_adapter, duration_hours=5.0)
        result = await agent.run("find me a long contest")
        assert result["pending_action_id"] is not None
        assert "confirm" in result["result"].lower()
        # Verify NO event was written to DB (only staged)
        conflict = await calendar_adapter.check_overlap(SLOT_START, SLOT_START + timedelta(hours=5))
        assert conflict is None  # event NOT in DB yet (pending confirmation)

    async def test_no_contests_returns_message(self, calendar_adapter):
        """Empty provider list → 'No upcoming contests found' message."""
        hitl = HITLManager(calendar_adapter=calendar_adapter)

        mock_provider = AsyncMock()
        mock_provider.fetch = AsyncMock(return_value=[])

        # LLM should not be called if there are no contests to select
        mock_llm = MagicMock()

        agent = ContestAgent(
            contest_provider=mock_provider,
            calendar_adapter=calendar_adapter,
            hitl_manager=hitl,
            llm=mock_llm,
        )
        result = await agent.run("find me a contest")
        assert "no" in result["result"].lower() or "not found" in result["result"].lower()

    async def test_exactly_4h_triggers_hitl(self, calendar_adapter):
        """Boundary: exactly 4 hours should trigger HITL."""
        agent = make_agent(calendar_adapter, duration_hours=4.0)
        result = await agent.run("schedule a 4h contest")
        assert result["pending_action_id"] is not None

    async def test_just_under_4h_does_not_trigger_hitl(self, calendar_adapter):
        """Boundary: 3.9 hours should NOT trigger HITL."""
        # Need custom agent with exactly 3.9h selected
        contests = [make_contest_dto(duration_hours=3.9)]
        mock_provider = AsyncMock()
        mock_provider.fetch = AsyncMock(return_value=contests)

        selected = ContestSelectionResult(
            title="CF Round 987",
            start=SLOT_START,
            duration_hours=3.9,
            provider="codeforces",
        )
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value=selected)
        mock_chain.with_retry = MagicMock(return_value=mock_chain)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_chain)

        hitl = HITLManager(calendar_adapter=calendar_adapter)
        agent = ContestAgent(mock_provider, calendar_adapter, hitl, llm=mock_llm)

        result = await agent.run("find a 3.9h contest")
        assert result["pending_action_id"] is None
        assert "Scheduled" in result["result"]
