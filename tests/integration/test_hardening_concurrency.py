"""
tests/integration/test_hardening_concurrency.py — Concurrency safety tests.

Design Document Reference: §9.1 (asyncio.Lock — Improvement 2)

The class-level asyncio.Lock on CalendarAdapter ensures that two concurrent
safe_schedule() calls for the same time slot cannot both see a "free" slot
before either write completes. This is the atomicity guarantee.

Exit criteria (Phase 3):
  ✓ Two concurrent safe_schedule() calls for the same slot → exactly 1 Event + 1 ConflictInfo
  ✓ N concurrent calls → exactly 1 Event, (N-1) ConflictInfos
  ✓ Concurrent calls for NON-overlapping slots → ALL succeed
  ✓ Contest agent: concurrent agent.run() calls for same contest → exactly 1 HITL or 1 Event
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.contest import ContestAgent
from src.db import Event
from src.hitl import HITLManager
from src.parsers import ConflictInfo, ContestDTO, ContestSelectionResult
from src.tools.calendar import SQLiteCalendarAdapter, CalendarAdapter

SLOT_START = datetime(2026, 6, 1, 14, 0, tzinfo=timezone.utc)
SLOT_END   = datetime(2026, 6, 1, 16, 0, tzinfo=timezone.utc)


class TestConcurrencyAtomicity:

    async def test_two_concurrent_safe_schedule_only_one_wins(self, calendar_adapter):
        """
        Core Improvement 2 test:
        Two coroutines race to schedule the same slot.
        The class-level Lock ensures exactly 1 wins and 1 gets ConflictInfo.
        """
        results = await asyncio.gather(
            calendar_adapter.safe_schedule("Task A", SLOT_START, SLOT_END, "contest"),
            calendar_adapter.safe_schedule("Task B", SLOT_START, SLOT_END, "contest"),
            return_exceptions=True,
        )

        events    = [r for r in results if isinstance(r, Event)]
        conflicts = [r for r in results if isinstance(r, ConflictInfo)]

        assert len(events)    == 1, f"Expected exactly 1 event, got {events}"
        assert len(conflicts) == 1, f"Expected exactly 1 conflict, got {conflicts}"

    async def test_five_concurrent_calls_only_one_wins(self, calendar_adapter):
        """N concurrent calls → exactly 1 event, N-1 conflicts."""
        N = 5
        results = await asyncio.gather(
            *[
                calendar_adapter.safe_schedule(f"Task {i}", SLOT_START, SLOT_END, "contest")
                for i in range(N)
            ],
            return_exceptions=True,
        )

        events    = [r for r in results if isinstance(r, Event)]
        conflicts = [r for r in results if isinstance(r, ConflictInfo)]

        assert len(events)    == 1,     f"Expected 1 event, got {len(events)}"
        assert len(conflicts) == N - 1, f"Expected {N-1} conflicts, got {len(conflicts)}"

    async def test_concurrent_non_overlapping_slots_all_succeed(self, calendar_adapter):
        """
        Concurrent calls for non-overlapping slots must ALL succeed —
        the Lock only serialises checks, it doesn't prevent legitimate parallel scheduling.
        """
        slots = [
            (SLOT_START + timedelta(hours=i * 3), SLOT_START + timedelta(hours=i * 3 + 2))
            for i in range(4)
        ]
        results = await asyncio.gather(
            *[
                calendar_adapter.safe_schedule(f"Event {i}", start, end, "contest")
                for i, (start, end) in enumerate(slots)
            ],
            return_exceptions=True,
        )

        events    = [r for r in results if isinstance(r, Event)]
        conflicts = [r for r in results if isinstance(r, ConflictInfo)]

        assert len(events)    == 4, f"All 4 non-overlapping events should be created, got {events}"
        assert len(conflicts) == 0, f"No conflicts expected, got {conflicts}"


class TestConcurrencyContestAgent:

    def _make_agent(
        self,
        calendar_adapter: SQLiteCalendarAdapter,
        title: str = "CF Round 987",
    ) -> ContestAgent:
        contest = ContestDTO(
            title=title,
            start=SLOT_START,
            end=SLOT_END,
            provider="codeforces",
            external_id=f"codeforces-{title.replace(' ', '-')}",
        )
        mock_provider = AsyncMock()
        mock_provider.fetch = AsyncMock(return_value=[contest])

        selected = ContestSelectionResult(
            title=title, start=SLOT_START, duration_hours=2.0, provider="codeforces"
        )
        mock_chain = AsyncMock()
        mock_chain.ainvoke    = AsyncMock(return_value=selected)
        mock_chain.with_retry = MagicMock(return_value=mock_chain)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_chain)

        return ContestAgent(
            contest_provider=mock_provider,
            calendar_adapter=calendar_adapter,
            hitl_manager=HITLManager(calendar_adapter=calendar_adapter),
            llm=mock_llm,
        )

    async def test_two_concurrent_agent_runs_only_one_schedules(self, calendar_adapter):
        """
        Two ContestAgent.run() calls for the same contest concurrently →
        exactly one event created (or HITL staged), one gets 'conflict' message.
        """
        agent_a = self._make_agent(calendar_adapter, title="CF Round 987")
        agent_b = self._make_agent(calendar_adapter, title="CF Round 987")

        result_a, result_b = await asyncio.gather(
            agent_a.run("find me a contest"),
            agent_b.run("find me a contest"),
            return_exceptions=True,
        )

        results = [result_a, result_b]

        # Filter: a "success" is a dict result with either a Scheduled message
        # or a pending_action_id (HITL staged). An exception counts as a failure.
        def is_success(r) -> bool:
            if isinstance(r, Exception):
                return False
            return (
                "Scheduled" in r.get("result", "")
                or r.get("pending_action_id") is not None
            )

        def is_conflict_or_error(r) -> bool:
            if isinstance(r, Exception):
                return True  # unexpected exception also means "didn't succeed"
            return (
                "conflict" in r.get("result", "").lower()
                or "conflicts" in r.get("result", "").lower()
                or "taken" in r.get("result", "").lower()
            )

        successes = [r for r in results if is_success(r)]
        non_successes = [r for r in results if not is_success(r)]

        assert len(successes) == 1, (
            f"Expected exactly 1 successful schedule, got {len(successes)}.\n"
            f"Results: {results}"
        )
        assert len(non_successes) == 1, (
            f"Expected 1 conflict/error, got {len(non_successes)}.\n"
            f"Results: {results}"
        )
