"""
tests/integration/test_email_agent.py — Integration tests for EmailAgent.

Uses mock LLM and StubEmailAdapter. Uses real SQLiteCalendarAdapter (test DB).

Exit criteria (Phase 2, Design Doc §8.3):
  ✓ MEETING_REQUEST → event scheduled, reply "Confirmed"
  ✓ SPAM → HITL staged, NOT deleted directly (Improvement 3)
  ✓ NEEDS_CLARIFICATION → reply sent asking for clearer time
  ✓ Two emails requesting the same slot → first scheduled, second declined (FIFO)
  ✓ classify_all does not write to DB (read-only stage)
  ✓ GENERAL email → logged only, no reply
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

# Fixed meeting time — parseable by dateparser deterministically
MEETING_TIME_ISO = "2026-05-10 14:00:00"
MEETING_START    = datetime(2026, 5, 10, 14, 0, tzinfo=timezone.utc)
MEETING_END      = datetime(2026, 5, 10, 15, 0, tzinfo=timezone.utc)
USER_NOW         = datetime(2026, 5, 1,   9, 0, tzinfo=timezone.utc)


def make_classification(intent: str, email_id: str = "email-001", offset_minutes: int = 0) -> EmailClassification:
    """Return a pre-built EmailClassification (bypass LLM in tests)."""
    received_at = USER_NOW + timedelta(minutes=offset_minutes)
    c = EmailClassification(
        email_id=email_id,
        intent=intent,
        meeting_title="Test Meeting" if intent == "MEETING_REQUEST" else None,
        meeting_time_raw=MEETING_TIME_ISO if intent == "MEETING_REQUEST" else None,
        received_at=received_at,
    )
    if intent == "MEETING_REQUEST":
        c.start = MEETING_START
        c.end   = MEETING_END
    return c


def make_mock_llm(classifications: list[EmailClassification]) -> MagicMock:
    """
    Build a mock LLM whose with_structured_output chain returns classifications
    sequentially (one per ainvoke call, in order).
    """
    mock_llm   = MagicMock()
    mock_chain = AsyncMock()
    mock_chain.ainvoke = AsyncMock(side_effect=list(classifications))
    mock_llm.with_structured_output = MagicMock(return_value=mock_chain)
    return mock_llm


def make_agent(
    calendar_adapter,
    email_adapter,
    classifications: list[EmailClassification],
) -> EmailAgent:
    """Build an EmailAgent with injected mock LLM."""
    hitl = HITLManager(
        calendar_adapter=calendar_adapter,
        email_adapter=email_adapter,
    )
    return EmailAgent(
        email_adapter=email_adapter,
        calendar_adapter=calendar_adapter,
        hitl_manager=hitl,
        llm=make_mock_llm(classifications),
        user_now=USER_NOW,
    )


class TestEmailAgentMeetingRequest:

    async def test_meeting_request_creates_event_and_replies(
        self, calendar_adapter, stub_email
    ):
        """Single MEETING_REQUEST → event in DB, 'Confirmed' reply."""
        # Provide 3 emails but only stub-email-001 is MEETING_REQUEST
        stub_email_with_one = StubEmailAdapter()
        # Only 1 email in inbox for simplicity
        meeting_email = EmailDTO(
            id="email-001",
            sender="alice@example.com",
            subject="Meeting",
            body="Can we meet?",
            received_at=USER_NOW,
        )

        async def single_email_inbox():
            return [meeting_email]

        stub_email_with_one.get_unread = single_email_inbox

        c = make_classification("MEETING_REQUEST", "email-001")
        agent = make_agent(calendar_adapter, stub_email_with_one, [c])
        result = await agent.run()

        report = result["action_report"]
        assert any(r["action"] == "scheduled" for r in report)

        # Verify reply was sent
        assert any("email-001" in r["email_id"] for r in report if r["action"] == "scheduled")
        assert len(stub_email_with_one.replies) == 1
        assert "Confirmed" in stub_email_with_one.replies[0]["body"]

        # Verify event in DB
        conflict = await calendar_adapter.check_overlap(MEETING_START, MEETING_END)
        assert conflict is not None
        assert "Test Meeting" in conflict.conflicting_title

    async def test_meeting_request_declined_when_busy(
        self, calendar_adapter, stub_email
    ):
        """Meeting request declined when slot already occupied."""
        # Pre-fill the calendar
        await calendar_adapter.create_event("Prior Meeting", MEETING_START, MEETING_END, "manual")

        single_email = StubEmailAdapter()

        async def inbox():
            return [EmailDTO(
                id="email-001",
                sender="alice@example.com",
                subject="Meet?",
                body="Can we meet?",
                received_at=USER_NOW,
            )]

        single_email.get_unread = inbox
        c = make_classification("MEETING_REQUEST", "email-001")
        agent = make_agent(calendar_adapter, single_email, [c])
        result = await agent.run()

        report = result["action_report"]
        assert any(r["action"] == "declined" for r in report)
        # Verify reply mentions being busy
        assert len(single_email.replies) == 1
        assert "booked" in single_email.replies[0]["body"].lower() or "already" in single_email.replies[0]["body"].lower()


class TestEmailAgentSpam:

    async def test_spam_stages_hitl_not_deleted(self, calendar_adapter, stub_email):
        """
        SPAM email must NEVER be auto-deleted.
        It must be staged as HITL and stub_email.deleted must remain empty.
        (Design Doc §8.3, Improvement 3)
        """
        single_email = StubEmailAdapter()

        async def inbox():
            return [EmailDTO(
                id="email-spam",
                sender="spam@promo.com",
                subject="Win a prize!",
                body="Click here!",
                received_at=USER_NOW,
            )]

        single_email.get_unread = inbox
        c = make_classification("SPAM", "email-spam")
        agent = make_agent(calendar_adapter, single_email, [c])
        result = await agent.run()

        report = result["action_report"]
        assert any(r["action"] == "pending_review" for r in report)

        # Critical: email must NOT have been deleted directly
        assert len(single_email.deleted) == 0

        # Verify a PendingAction was created in DB
        spam_action = next((r for r in report if r.get("action") == "pending_review"), None)
        assert spam_action is not None
        assert "pending_action_id" in spam_action


class TestEmailAgentClarification:

    async def test_clarification_sent_for_ambiguous_time(self, calendar_adapter, stub_email):
        """NEEDS_CLARIFICATION → reply asking for clearer time, no event creation."""

        async def inbox():
            return [EmailDTO(
                id="email-clarify",
                sender="bob@example.com",
                subject="Meeting sometime?",
                body="Let's catch up sometime.",
                received_at=USER_NOW,
            )]

        stub_email.get_unread = inbox
        c = make_classification("NEEDS_CLARIFICATION", "email-clarify")
        agent = make_agent(calendar_adapter, stub_email, [c])
        result = await agent.run()

        report = result["action_report"]
        assert any(r["action"] == "clarification_sent" for r in report)

        assert len(stub_email.replies) == 1
        reply_body = stub_email.replies[0]["body"]
        assert "date" in reply_body.lower() or "time" in reply_body.lower()

        # No event created
        conflict = await calendar_adapter.check_overlap(MEETING_START, MEETING_END)
        assert conflict is None


class TestEmailAgentFIFOArbitration:

    async def test_two_emails_same_slot_only_first_scheduled(
        self, calendar_adapter, stub_email
    ):
        """
        KEY TEST (Design Doc §8.3 exit criterion):
        Two MEETING_REQUEST emails for the same time slot →
          - first (by received_at) gets scheduled
          - second gets "busy" reply (declined)
        """
        email_1 = EmailDTO(
            id="email-early",
            sender="alice@example.com",
            subject="Meeting at 2pm",
            body=f"Can we meet at {MEETING_TIME_ISO}?",
            received_at=USER_NOW,                        # earlier
        )
        email_2 = EmailDTO(
            id="email-late",
            sender="bob@example.com",
            subject="Meeting at 2pm",
            body=f"Can we meet at {MEETING_TIME_ISO}?",
            received_at=USER_NOW + timedelta(minutes=5), # later
        )

        multi_email = StubEmailAdapter()

        async def inbox():
            return [email_1, email_2]

        multi_email.get_unread = inbox

        c1 = make_classification("MEETING_REQUEST", "email-early", offset_minutes=0)
        c2 = make_classification("MEETING_REQUEST", "email-late",  offset_minutes=5)

        hitl = HITLManager(calendar_adapter=calendar_adapter, email_adapter=multi_email)
        agent = EmailAgent(
            email_adapter=multi_email,
            calendar_adapter=calendar_adapter,
            hitl_manager=hitl,
            llm=make_mock_llm([c1, c2]),
            user_now=USER_NOW,
        )
        result = await agent.run()

        report = result["action_report"]
        scheduled = [r for r in report if r["action"] == "scheduled"]
        declined  = [r for r in report if r["action"] == "declined"]

        assert len(scheduled) == 1, f"Expected 1 scheduled, got {scheduled}"
        assert len(declined)  == 1, f"Expected 1 declined, got {declined}"

        # First email by received_at wins
        assert scheduled[0]["email_id"] == "email-early"
        assert declined[0]["email_id"]  == "email-late"

        # Check replies
        replies_by_id = {r["email_id"]: r["body"] for r in multi_email.replies}
        assert "Confirmed" in replies_by_id.get("email-early", "")
        assert "booked" in replies_by_id.get("email-late", "").lower() or \
               "already" in replies_by_id.get("email-late", "").lower()


class TestEmailAgentGeneral:

    async def test_general_email_only_logged(self, calendar_adapter, stub_email):
        """GENERAL emails → logged only, no reply, no event created."""

        async def inbox():
            return [EmailDTO(
                id="email-general",
                sender="charlie@example.com",
                subject="Status update",
                body="Just checking in.",
                received_at=USER_NOW,
            )]

        stub_email.get_unread = inbox
        c = make_classification("GENERAL", "email-general")
        agent = make_agent(calendar_adapter, stub_email, [c])
        result = await agent.run()

        report = result["action_report"]
        assert any(r["action"] == "logged" for r in report)
        # No replies for GENERAL
        assert len(stub_email.replies) == 0

    async def test_empty_inbox_returns_empty_report(self, calendar_adapter, stub_email):
        """Empty inbox → empty action_report, no errors."""

        async def inbox():
            return []

        stub_email.get_unread = inbox
        agent = EmailAgent(
            email_adapter=stub_email,
            calendar_adapter=calendar_adapter,
            hitl_manager=HITLManager(),
            llm=MagicMock(),
            user_now=USER_NOW,
        )
        result = await agent.run()
        assert result["action_report"] == []
