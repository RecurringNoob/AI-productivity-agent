"""
tests/unit/test_parsers.py — Unit tests for all Pydantic schemas in src/parsers.py

Validates that:
  - LLM output schemas enforce their Literal constraints
  - Field validators (ge/le on confidence) work correctly
  - DTO fields are accepted and populated correctly
  - Invalid inputs raise ValidationError
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.parsers import (
    ConflictInfo,
    ContestDTO,
    ContestSelectionResult,
    EmailClassification,
    EmailDTO,
    RoutingDecision,
)


# ---------------------------------------------------------------------------
# RoutingDecision
# ---------------------------------------------------------------------------


class TestRoutingDecision:

    def test_valid_contest_agent(self):
        rd = RoutingDecision(agent="CONTEST_AGENT", confidence=0.9)
        assert rd.agent == "CONTEST_AGENT"
        assert rd.confidence == 0.9

    def test_valid_email_agent(self):
        rd = RoutingDecision(agent="EMAIL_AGENT", confidence=0.75)
        assert rd.agent == "EMAIL_AGENT"

    def test_valid_general(self):
        rd = RoutingDecision(agent="GENERAL", confidence=0.5)
        assert rd.agent == "GENERAL"

    def test_confidence_boundary_zero(self):
        rd = RoutingDecision(agent="GENERAL", confidence=0.0)
        assert rd.confidence == 0.0

    def test_confidence_boundary_one(self):
        rd = RoutingDecision(agent="GENERAL", confidence=1.0)
        assert rd.confidence == 1.0

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValidationError):
            RoutingDecision(agent="GENERAL", confidence=1.1)

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValidationError):
            RoutingDecision(agent="GENERAL", confidence=-0.1)

    def test_invalid_agent_raises(self):
        with pytest.raises(ValidationError):
            RoutingDecision(agent="UNKNOWN_AGENT", confidence=0.9)

    def test_missing_agent_raises(self):
        with pytest.raises(ValidationError):
            RoutingDecision(confidence=0.9)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ContestSelectionResult
# ---------------------------------------------------------------------------


class TestContestSelectionResult:

    def test_valid_result(self):
        now = datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc)
        csr = ContestSelectionResult(
            title="CF Round 987",
            start=now,
            duration_hours=2.5,
            provider="codeforces",
        )
        assert csr.title == "CF Round 987"
        assert csr.duration_hours == 2.5
        assert csr.provider == "codeforces"

    def test_missing_required_field_raises(self):
        now = datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc)
        with pytest.raises(ValidationError):
            ContestSelectionResult(title="CF Round 987", start=now, duration_hours=2.0)  # missing provider


# ---------------------------------------------------------------------------
# EmailClassification
# ---------------------------------------------------------------------------


class TestEmailClassification:

    def test_valid_spam(self):
        ec = EmailClassification(intent="SPAM")
        assert ec.intent == "SPAM"

    def test_valid_meeting_request(self):
        ec = EmailClassification(
            intent="MEETING_REQUEST",
            meeting_title="Sync Meeting",
            meeting_time_raw="tomorrow at 2pm",
        )
        assert ec.meeting_title == "Sync Meeting"
        assert ec.meeting_time_raw == "tomorrow at 2pm"

    def test_default_travel_buffer_minutes(self):
        """Default buffer is 15 per Design Doc §6.2 (Improvement 4)."""
        ec = EmailClassification(intent="GENERAL")
        assert ec.travel_buffer_minutes == 15

    def test_custom_travel_buffer_minutes(self):
        ec = EmailClassification(intent="MEETING_REQUEST", travel_buffer_minutes=30)
        assert ec.travel_buffer_minutes == 30

    def test_needs_clarification_intent(self):
        ec = EmailClassification(intent="NEEDS_CLARIFICATION")
        assert ec.intent == "NEEDS_CLARIFICATION"

    def test_invalid_intent_raises(self):
        with pytest.raises(ValidationError):
            EmailClassification(intent="DELETE_ALL")

    def test_optional_fields_default_none(self):
        ec = EmailClassification(intent="SPAM")
        assert ec.email_id is None
        assert ec.meeting_title is None
        assert ec.meeting_time_raw is None
        assert ec.received_at is None
        assert ec.start is None
        assert ec.end is None


# ---------------------------------------------------------------------------
# ContestDTO
# ---------------------------------------------------------------------------


class TestContestDTO:

    def test_valid_contest_dto(self):
        now = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)
        dto = ContestDTO(
            title="CF Round 987",
            start=now,
            end=now,
            provider="codeforces",
            external_id="codeforces-987",
        )
        assert dto.external_id == "codeforces-987"
        assert dto.provider == "codeforces"


# ---------------------------------------------------------------------------
# EmailDTO
# ---------------------------------------------------------------------------


class TestEmailDTO:

    def test_valid_email_dto(self):
        now = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)
        dto = EmailDTO(
            id="email-001",
            sender="alice@example.com",
            subject="Meeting",
            body="Let's meet tomorrow",
            received_at=now,
        )
        assert dto.id == "email-001"
        assert dto.received_at == now


# ---------------------------------------------------------------------------
# ConflictInfo
# ---------------------------------------------------------------------------


class TestConflictInfo:

    def test_valid_conflict_info(self):
        now = datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc)
        ci = ConflictInfo(
            conflicting_title="Existing Meeting",
            conflicting_start=now,
            conflicting_end=now,
            buffer_applied=20,
        )
        assert ci.conflicting_title == "Existing Meeting"
        assert ci.buffer_applied == 20

    def test_buffer_applied_zero(self):
        now = datetime(2026, 5, 1, 14, 0, tzinfo=timezone.utc)
        ci = ConflictInfo(
            conflicting_title="X",
            conflicting_start=now,
            conflicting_end=now,
            buffer_applied=0,
        )
        assert ci.buffer_applied == 0
