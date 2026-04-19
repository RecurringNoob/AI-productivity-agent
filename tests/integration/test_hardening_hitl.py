"""
tests/integration/test_hardening_hitl.py — HITL end-to-end workflow tests.

Design Document Reference: §9.5 (HITL Manager), Improvement 3

Tests the full HITL lifecycle:
  stage → confirm("confirm") → _execute() → event in DB
  stage → confirm("undo")    → no event, status CANCELLED
  stage → expire_stale()     → status EXPIRED, confirm raises

Also tests:
  ✓ ContestAgent HITL flow (≥4h → staged → confirm → event)
  ✓ EmailAgent SPAM HITL flow (spam → staged → confirm → email deleted)
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.contest import ContestAgent
from src.agents.email import EmailAgent
from src.db import PendingAction
from src.hitl import HITLManager
from src.parsers import ContestDTO, ContestSelectionResult, EmailClassification, EmailDTO
from src.tools.calendar import SQLiteCalendarAdapter
from src.tools.email import StubEmailAdapter

SLOT_START = datetime(2026, 6, 10, 14, 0, tzinfo=timezone.utc)
SLOT_END   = datetime(2026, 6, 10, 19, 0, tzinfo=timezone.utc)  # 5 hours (> 4h → HITL)
USER_NOW   = datetime(2026, 6, 1, 9, 0, tzinfo=timezone.utc)


class TestHITLContestWorkflow:

    def _make_5h_contest_agent(self, calendar_adapter):
        contest = ContestDTO(
            title="ICPC World Finals",
            start=SLOT_START,
            end=SLOT_END,
            provider="codeforces",
            external_id="codeforces-icpc-2026",
        )
        mock_provider = AsyncMock()
        mock_provider.fetch = AsyncMock(return_value=[contest])

        selected = ContestSelectionResult(
            title="ICPC World Finals",
            start=SLOT_START,
            duration_hours=5.0,
            provider="codeforces",
        )
        mock_chain = AsyncMock()
        mock_chain.ainvoke    = AsyncMock(return_value=selected)
        mock_chain.with_retry = MagicMock(return_value=mock_chain)
        mock_llm = MagicMock()
        mock_llm.with_structured_output = MagicMock(return_value=mock_chain)

        hitl = HITLManager(calendar_adapter=calendar_adapter)
        return ContestAgent(
            contest_provider=mock_provider,
            calendar_adapter=calendar_adapter,
            hitl_manager=hitl,
            llm=mock_llm,
        ), hitl

    async def test_contest_hitl_stage_then_confirm_creates_event(self, calendar_adapter):
        """Full HITL path: run agent → stage → user confirms → event created."""
        agent, hitl = self._make_5h_contest_agent(calendar_adapter)
        result = await agent.run("schedule ICPC finals")

        pending_id = result["pending_action_id"]
        assert pending_id is not None

        # No event in DB yet (pending confirmation)
        assert await calendar_adapter.check_overlap(SLOT_START, SLOT_END) is None

        # User confirms
        msg = await hitl.confirm(pending_id, "confirm")
        assert "ICPC World Finals" in msg

        # Event now in DB
        conflict = await calendar_adapter.check_overlap(SLOT_START, SLOT_END)
        assert conflict is not None
        assert "ICPC World Finals" in conflict.conflicting_title

    async def test_contest_hitl_stage_then_undo_no_event(self, calendar_adapter):
        """Full HITL path: run agent → stage → user cancels → no event."""
        agent, hitl = self._make_5h_contest_agent(calendar_adapter)
        result = await agent.run("schedule ICPC finals")

        pending_id = result["pending_action_id"]
        assert pending_id is not None

        await hitl.confirm(pending_id, "undo")

        # No event in DB
        assert await calendar_adapter.check_overlap(SLOT_START, SLOT_END) is None

    async def test_expired_hitl_raises_on_confirm(self, calendar_adapter):
        """Expired action → confirm raises ValueError before executing."""
        import json
        from uuid import uuid4
        from src.db import get_async_session

        # Directly insert expired pending action
        action = PendingAction(
            id=str(uuid4()),
            action_type="schedule_event",
            payload=json.dumps({
                "title": "Ghost Event",
                "start": SLOT_START.isoformat(),
                "end":   SLOT_END.isoformat(),
                "source": "contest",
            }),
            description="Expired contest",
            status="REQUIRES_REVIEW",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        async with get_async_session() as session:
            session.add(action)
            await session.commit()

        hitl = HITLManager(calendar_adapter=calendar_adapter)
        with pytest.raises(ValueError, match="expired"):
            await hitl.confirm(action.id, "confirm")

        # No event created
        assert await calendar_adapter.check_overlap(SLOT_START, SLOT_END) is None

    async def test_double_confirm_raises_second_time(self, calendar_adapter):
        """confirming twice → second raises ValueError (already CONFIRMED)."""
        agent, hitl = self._make_5h_contest_agent(calendar_adapter)
        result = await agent.run("schedule ICPC finals")
        pending_id = result["pending_action_id"]

        await hitl.confirm(pending_id, "confirm")

        with pytest.raises(ValueError, match="already in state"):
            await hitl.confirm(pending_id, "confirm")


class TestHITLEmailSpamWorkflow:

    async def test_spam_hitl_stage_then_confirm_deletes_email(
        self, calendar_adapter
    ):
        """Full SPAM HITL path: classify spam → stage → confirm → email deleted."""
        stub = StubEmailAdapter()
        spam_email = EmailDTO(
            id="email-spam-001",
            sender="spammer@promo.com",
            subject="Win money!",
            body="Click here now!",
            received_at=USER_NOW,
        )

        async def inbox():
            return [spam_email]

        stub.get_unread = inbox

        spam_classification = EmailClassification(
            email_id="email-spam-001",
            intent="SPAM",
            received_at=USER_NOW,
        )

        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value=spam_classification)
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
        spam_report = next(r for r in report if r["action"] == "pending_review")
        pending_id = spam_report["pending_action_id"]

        # Not deleted yet
        assert len(stub.deleted) == 0

        # Confirm deletion
        await hitl.confirm(pending_id, "confirm")
        assert "email-spam-001" in stub.deleted

    async def test_spam_undo_does_not_delete_email(self, calendar_adapter):
        """Undo the spam HITL → email NOT deleted."""
        stub = StubEmailAdapter()
        spam_email = EmailDTO(
            id="email-spam-002",
            sender="spammer@promo.com",
            subject="Promo",
            body="Click!",
            received_at=USER_NOW,
        )

        async def inbox():
            return [spam_email]

        stub.get_unread = inbox

        spam_classification = EmailClassification(
            email_id="email-spam-002",
            intent="SPAM",
            received_at=USER_NOW,
        )
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value=spam_classification)
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
        pending_id = next(r for r in report if r["action"] == "pending_review")["pending_action_id"]

        await hitl.confirm(pending_id, "undo")
        assert len(stub.deleted) == 0
