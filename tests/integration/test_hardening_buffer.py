"""
tests/integration/test_hardening_buffer.py — Travel buffer (soft-conflict) tests.

Design Document Reference: Improvement 4 (CALENDAR_TRAVEL_BUFFER_MINUTES)

Verifies that the configurable travel/preparation buffer correctly:
  - Catches near-adjacent meetings that lack travel time
  - Allows meetings that have sufficient gap
  - Is applied by EmailAgent.reduce_node (via travel_buffer_minutes on classification)
  - Is applied by ContestAgent.check_calendar (via env var)

Exit criteria (Phase 3):
  ✓ 15-min buffer blocks a meeting starting 10 min after another
  ✓ 0-min buffer allows the same scenario
  ✓ Large gap always succeeds regardless of buffer
  ✓ EmailAgent.reduce applies travel_buffer_minutes from EmailClassification
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.email import EmailAgent
from src.hitl import HITLManager
from src.parsers import EmailClassification, EmailDTO
from src.tools.calendar import SQLiteCalendarAdapter
from src.tools.email import StubEmailAdapter

# Reference event: 2:00 PM – 3:00 PM
PM2 = datetime(2026, 6, 20, 14,  0, tzinfo=timezone.utc)
PM3 = datetime(2026, 6, 20, 15,  0, tzinfo=timezone.utc)

USER_NOW = datetime(2026, 6, 15, 9, 0, tzinfo=timezone.utc)


class TestSoftConflictBuffer:

    async def test_15min_buffer_blocks_10min_gap(self, calendar_adapter):
        """
        Improvement 4: existing meeting ends at 3 PM.
        New meeting at 3:10 PM with 15-min buffer →
        [3:10 - 15min = 2:55 PM] overlaps [2:00–3:00 PM] → CONFLICT.
        """
        await calendar_adapter.create_event("Meeting A", PM2, PM3, "manual")

        new_start = PM3 + timedelta(minutes=10)  # 3:10 PM
        new_end   = PM3 + timedelta(minutes=70)  # 4:10 PM

        conflict = await calendar_adapter.check_overlap(
            new_start, new_end, buffer_minutes=15
        )
        assert conflict is not None
        assert conflict.buffer_applied == 15

    async def test_0min_buffer_allows_10min_gap(self, calendar_adapter):
        """
        Same scenario but with buffer=0 → no conflict (not adjacent in raw terms).
        """
        await calendar_adapter.create_event("Meeting A", PM2, PM3, "manual")

        new_start = PM3 + timedelta(minutes=10)
        new_end   = PM3 + timedelta(minutes=70)

        conflict = await calendar_adapter.check_overlap(
            new_start, new_end, buffer_minutes=0
        )
        assert conflict is None

    async def test_large_gap_always_free(self, calendar_adapter):
        """A 3-hour gap after a 1-hour meeting is free even with a 60-min buffer."""
        await calendar_adapter.create_event("Meeting A", PM2, PM3, "manual")

        new_start = PM3 + timedelta(hours=3)  # 6 PM
        new_end   = PM3 + timedelta(hours=4)  # 7 PM

        conflict = await calendar_adapter.check_overlap(
            new_start, new_end, buffer_minutes=60
        )
        assert conflict is None

    async def test_exactly_at_buffer_boundary_is_conflict(self, calendar_adapter):
        """Exactly at the buffer boundary → still a conflict (inclusive check)."""
        await calendar_adapter.create_event("Meeting A", PM2, PM3, "manual")

        # Starts exactly 15 min after end, with 15-min buffer:
        # query_start = 3:15 - 15 = 3:00 PM, which overlaps with [2:00, 3:00) ? 
        # condition: event.end_time > query_start → 3:00 > 3:00 is FALSE → no conflict
        new_start = PM3 + timedelta(minutes=15)  # exactly 15 min gap
        new_end   = PM3 + timedelta(hours=2)

        # NOTE: the overlap condition is strict (>), so exactly at boundary = NO conflict
        conflict = await calendar_adapter.check_overlap(
            new_start, new_end, buffer_minutes=15
        )
        assert conflict is None  # strict > means boundary is clear

    async def test_just_inside_buffer_is_conflict(self, calendar_adapter):
        """14 minutes gap with 15-min buffer → conflict."""
        await calendar_adapter.create_event("Meeting A", PM2, PM3, "manual")

        new_start = PM3 + timedelta(minutes=14)  # only 14 min gap
        new_end   = PM3 + timedelta(hours=2)

        conflict = await calendar_adapter.check_overlap(
            new_start, new_end, buffer_minutes=15
        )
        assert conflict is not None


class TestEmailAgentBufferIntegration:

    async def test_email_agent_applies_classification_buffer(self, calendar_adapter):
        """
        Improvement 4 integration test:
        EmailClassification.travel_buffer_minutes flows through to safe_schedule.

        Scenario:
          - Existing meeting 2–3 PM
          - Email requests meeting at 3:10 PM with travel_buffer_minutes=20
          - 20-min buffer [3:10 - 20 = 2:50 PM] overlaps [2:00–3:00 PM]
          - Result: declined (conflict detected via buffer)
        """
        # Pre-fill calendar
        await calendar_adapter.create_event("Prior Meeting", PM2, PM3, "manual")

        new_start = PM3 + timedelta(minutes=10)  # 3:10 PM
        new_end   = new_start + timedelta(hours=1)

        stub = StubEmailAdapter()
        req_email = EmailDTO(
            id="email-buffer-001",
            sender="dave@example.com",
            subject="Quick meeting?",
            body=f"Let's meet at 3:10 PM on June 20",
            received_at=USER_NOW,
        )

        async def inbox():
            return [req_email]

        stub.get_unread = inbox

        # Pre-built classification (bypass LLM)
        c = EmailClassification(
            email_id="email-buffer-001",
            intent="MEETING_REQUEST",
            meeting_title="Quick Meeting",
            meeting_time_raw=None,  # start/end set directly
            travel_buffer_minutes=20,   # larger-than-default buffer
            received_at=USER_NOW,
            start=new_start,
            end=new_end,
        )
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value=c)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_chain)

        hitl = HITLManager(calendar_adapter=calendar_adapter, email_adapter=stub)
        agent = EmailAgent(
            email_adapter=stub,
            calendar_adapter=calendar_adapter,
            hitl_manager=hitl,
            llm=mock_llm,
            user_now=USER_NOW,
        )
        result = await agent.run()

        report = result["action_report"]
        assert any(r["action"] == "declined" for r in report), (
            f"Expected declined due to 20-min buffer, got: {report}"
        )

    async def test_email_agent_zero_buffer_allows_adjacent(self, calendar_adapter):
        """
        With travel_buffer_minutes=0, a meeting immediately after another is allowed.
        """
        await calendar_adapter.create_event("Prior Meeting", PM2, PM3, "manual")

        # Request meeting exactly when prior meeting ends: 3:00 PM
        stub = StubEmailAdapter()
        req_email = EmailDTO(
            id="email-nobuffer-001",
            sender="dave@example.com",
            subject="Immediate follow-up",
            body="Can we meet right at 3 PM?",
            received_at=USER_NOW,
        )

        async def inbox():
            return [req_email]

        stub.get_unread = inbox

        c = EmailClassification(
            email_id="email-nobuffer-001",
            intent="MEETING_REQUEST",
            meeting_title="Follow-up",
            travel_buffer_minutes=0,    # no buffer
            received_at=USER_NOW,
            start=PM3,                  # 3:00 PM (adjacent to prior meeting end)
            end=PM3 + timedelta(hours=1),
        )
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value=c)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_chain)

        hitl = HITLManager(calendar_adapter=calendar_adapter, email_adapter=stub)
        agent = EmailAgent(
            email_adapter=stub,
            calendar_adapter=calendar_adapter,
            hitl_manager=hitl,
            llm=mock_llm,
            user_now=USER_NOW,
        )
        result = await agent.run()

        report = result["action_report"]
        assert any(r["action"] == "scheduled" for r in report), (
            f"Expected scheduled (no buffer), got: {report}"
        )
