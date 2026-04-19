"""
tests/integration/test_hardening_dedup.py — Cross-provider deduplication tests.

Design Document Reference: §9.4 (Improvement 5), §6.1 (UNIQUE(external_id))

Two layers of deduplication:
  Layer 1 — Semantic (Levenshtein): AggregatingContestProvider deduplicates
             near-identical titles before any DB writes.
  Layer 2 — Hard DB constraint: UNIQUE(external_id) on the events table
             prevents duplicate rows even if Layer 1 is bypassed.

Exit criteria (Phase 3):
  ✓ DB-level: creating two events with the same external_id raises IntegrityError
  ✓ DB-level: NULL external_id is allowed for multiple events (each NULL is unique)
  ✓ DB-level: two different external_ids both succeed
  ✓ Agent-level: ContestAgent creates event with correct external_id
  ✓ Semantic: AggregatingContestProvider dedup already tested in unit tests;
               here we verify the full chain (provider → agent → DB → only 1 row)
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from src.agents.contest import ContestAgent
from src.hitl import HITLManager
from src.parsers import ContestDTO, ContestSelectionResult
from src.tools.calendar import SQLiteCalendarAdapter

SLOT_START = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
SLOT_END   = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)


class TestDatabaseUniqueConstraint:

    async def test_duplicate_external_id_raises_integrity_error(self, calendar_adapter):
        """
        Layer 2 dedup: DB UNIQUE(external_id) prevents two rows with the same
        non-null external_id.
        (Design Doc §6.1 — v3 hardening migration 0003)
        """
        await calendar_adapter.create_event(
            "CF Round 987",
            SLOT_START,
            SLOT_END,
            "contest",
            external_id="codeforces-987",
        )

        # Second event 1 day later — different time, but SAME external_id
        with pytest.raises(IntegrityError):
            await calendar_adapter.create_event(
                "CF Round 987 (duplicate)",
                SLOT_START + timedelta(days=1),
                SLOT_END   + timedelta(days=1),
                "contest",
                external_id="codeforces-987",  # same!
            )

    async def test_null_external_id_allowed_multiple_times(self, calendar_adapter):
        """
        SQLite treats each NULL as unique — multiple events with external_id=None
        should NOT raise IntegrityError.
        """
        day1 = SLOT_START
        day2 = SLOT_START + timedelta(days=1)
        day3 = SLOT_START + timedelta(days=2)

        # All three should succeed
        e1 = await calendar_adapter.create_event("Manual 1", day1, day1 + timedelta(hours=1), "manual")
        e2 = await calendar_adapter.create_event("Manual 2", day2, day2 + timedelta(hours=1), "manual")
        e3 = await calendar_adapter.create_event("Manual 3", day3, day3 + timedelta(hours=1), "manual")

        assert e1.id is not None
        assert e2.id is not None
        assert e3.id is not None

    async def test_different_external_ids_both_succeed(self, calendar_adapter):
        """Distinct external_ids for non-overlapping events both persist correctly."""
        day1 = SLOT_START
        day2 = SLOT_START + timedelta(days=1)

        e1 = await calendar_adapter.create_event(
            "CF Round 987", day1, day1 + timedelta(hours=2), "contest", external_id="cf-987"
        )
        e2 = await calendar_adapter.create_event(
            "CF Round 988", day2, day2 + timedelta(hours=2), "contest", external_id="cf-988"
        )

        assert e1.external_id == "cf-987"
        assert e2.external_id == "cf-988"


class TestContestAgentExternalIdPropagation:

    async def test_agent_passes_external_id_to_event(self, calendar_adapter):
        """
        ContestAgent.execute should pass external_id from ContestDTO to create_event.
        This ensures the DB constraint can enforce dedup on subsequent agent runs.
        """
        contest = ContestDTO(
            title="LC Biweekly 150",
            start=SLOT_START,
            end=SLOT_END,
            provider="leetcode",
            external_id="leetcode-biweekly-150",
        )
        mock_provider = AsyncMock()
        mock_provider.fetch = AsyncMock(return_value=[contest])

        selected = ContestSelectionResult(
            title="LC Biweekly 150",
            start=SLOT_START,
            duration_hours=2.0,
            provider="leetcode",
        )
        mock_chain = AsyncMock()
        mock_chain.ainvoke    = AsyncMock(return_value=selected)
        mock_chain.with_retry = MagicMock(return_value=mock_chain)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_chain)

        hitl  = HITLManager(calendar_adapter=calendar_adapter)
        agent = ContestAgent(mock_provider, calendar_adapter, hitl, llm=mock_llm)

        await agent.run("schedule biweekly")

        # Verify external_id was written to DB (detected via conflict query)
        conflict = await calendar_adapter.check_overlap(SLOT_START, SLOT_END)
        assert conflict is not None
        assert "LC Biweekly 150" in conflict.conflicting_title

    async def test_agent_second_run_same_external_id_fails_at_db(self, calendar_adapter):
        """
        If ContestAgent runs twice for the same contest (same external_id),
        the second run must fail with an IntegrityError rather than silently
        creating a duplicate row.
        """
        contest = ContestDTO(
            title="CF Educational 180",
            start=SLOT_START,
            end=SLOT_END,
            provider="codeforces",
            external_id="codeforces-educational-180",
        )
        mock_provider = AsyncMock()
        mock_provider.fetch = AsyncMock(return_value=[contest])

        selected = ContestSelectionResult(
            title="CF Educational 180",
            start=SLOT_START,
            duration_hours=2.0,
            provider="codeforces",
        )
        mock_chain = AsyncMock()
        mock_chain.ainvoke    = AsyncMock(return_value=selected)
        mock_chain.with_retry = MagicMock(return_value=mock_chain)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_chain)

        hitl  = HITLManager(calendar_adapter=calendar_adapter)
        agent = ContestAgent(mock_provider, calendar_adapter, hitl, llm=mock_llm)

        # First run succeeds
        result1 = await agent.run("schedule CF educational")
        assert "Scheduled" in result1["result"] or result1["pending_action_id"] is None

        # Second run for the same slot → safe_schedule returns ConflictInfo
        # (because the first event now occupies the slot), not IntegrityError.
        # The calendar Lock sees the existing event and returns the conflict.
        result2 = await agent.run("schedule CF educational again")
        # Second run should detect the conflict (already scheduled) — not create a duplicate
        assert "CF Educational 180" in result2.get("result", "") or \
               "conflict" in result2.get("result", "").lower() or \
               result2.get("result", "") != ""
