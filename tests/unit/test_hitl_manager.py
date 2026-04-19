"""
tests/unit/test_hitl_manager.py — Unit tests for src/hitl.py

Exit criteria (Phase 3, Design Doc §9.5, Improvement 3):
  ✓ stage() returns a UUID and persists a PendingAction in REQUIRES_REVIEW state
  ✓ stage() sets a correct expires_at
  ✓ stage() stores JSON-serialised payload
  ✓ confirm("confirm") executes action and marks CONFIRMED
  ✓ confirm("undo") marks CANCELLED without calling _execute
  ✓ confirm() raises ValueError for unknown id
  ✓ confirm() raises ValueError if action already CONFIRMED
  ✓ confirm() marks EXPIRED and raises ValueError for past expires_at
  ✓ expire_stale() bulk-expires past rows only
  ✓ expire_stale() returns correct count
  ✓ expire_stale() leaves future / already-resolved rows untouched
  ✓ _execute("schedule_event") creates a calendar event
  ✓ _execute("delete_email") calls email_adapter.delete
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from uuid import uuid4, UUID

import pytest
from sqlalchemy import select

from src.db import PendingAction, get_async_session
from src.hitl import HITLManager
from src.tools.calendar import SQLiteCalendarAdapter
from src.tools.email import StubEmailAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SLOT_START = datetime(2026, 5, 15, 14, 0, tzinfo=timezone.utc)
SLOT_END   = datetime(2026, 5, 15, 16, 0, tzinfo=timezone.utc)


async def _get_action(action_id: str) -> PendingAction | None:
    """Helper: fetch a PendingAction from the test DB."""
    async with get_async_session() as session:
        result = await session.execute(
            select(PendingAction).where(PendingAction.id == action_id)
        )
        return result.scalars().first()


async def _create_action(
    *,
    status: str = "REQUIRES_REVIEW",
    expires_offset_minutes: int = 30,
    action_type: str = "schedule_event",
    payload: dict | None = None,
) -> PendingAction:
    """Helper: directly persist a PendingAction for testing edge cases."""
    action = PendingAction(
        id=str(uuid4()),
        action_type=action_type,
        payload=json.dumps(payload or {"title": "Test Event",
                                       "start": SLOT_START.isoformat(),
                                       "end":   SLOT_END.isoformat()}),
        description="Test action",
        status=status,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=expires_offset_minutes),
    )
    async with get_async_session() as session:
        session.add(action)
        await session.commit()
    return action


# ===========================================================================
# stage()
# ===========================================================================


class TestHITLStage:

    async def test_stage_returns_string_uuid(self, calendar_adapter):
        hitl = HITLManager(calendar_adapter=calendar_adapter)
        action_id = await hitl.stage("schedule_event", {"title": "T"}, "Test")
        # Must be a valid UUID string
        UUID(action_id)  # raises ValueError if not valid UUID
        assert isinstance(action_id, str)

    async def test_stage_persists_requires_review(self, calendar_adapter):
        hitl = HITLManager(calendar_adapter=calendar_adapter)
        action_id = await hitl.stage(
            "schedule_event",
            {"title": "Test", "start": SLOT_START.isoformat(), "end": SLOT_END.isoformat()},
            "Schedule test event",
        )
        action = await _get_action(action_id)
        assert action is not None
        assert action.status == "REQUIRES_REVIEW"
        assert action.action_type == "schedule_event"

    async def test_stage_payload_is_json_serialised(self, calendar_adapter):
        hitl = HITLManager(calendar_adapter=calendar_adapter)
        payload = {"title": "CF Round", "start": "2026-05-01T14:00:00"}
        action_id = await hitl.stage("schedule_event", payload, "test")
        action = await _get_action(action_id)
        assert json.loads(action.payload) == payload

    async def test_stage_expires_at_in_future(self, calendar_adapter):
        hitl = HITLManager(calendar_adapter=calendar_adapter)
        before = datetime.now(timezone.utc)
        action_id = await hitl.stage("schedule_event", {}, "test")
        action = await _get_action(action_id)
        expires = (
            action.expires_at.replace(tzinfo=timezone.utc)
            if action.expires_at.tzinfo is None
            else action.expires_at
        )
        assert expires > before

    async def test_multiple_stages_create_separate_actions(self, calendar_adapter):
        hitl = HITLManager(calendar_adapter=calendar_adapter)
        id_1 = await hitl.stage("schedule_event", {}, "action 1")
        id_2 = await hitl.stage("schedule_event", {}, "action 2")
        assert id_1 != id_2


# ===========================================================================
# confirm()
# ===========================================================================


class TestHITLConfirm:

    async def test_confirm_marks_confirmed(self, calendar_adapter):
        hitl = HITLManager(calendar_adapter=calendar_adapter)
        action_id = await hitl.stage(
            "schedule_event",
            {"title": "My Event",
             "start": SLOT_START.isoformat(),
             "end":   SLOT_END.isoformat(),
             "source": "contest"},
            "Test",
        )
        await hitl.confirm(action_id, "confirm")
        action = await _get_action(action_id)
        assert action.status == "CONFIRMED"

    async def test_undo_marks_cancelled(self, calendar_adapter):
        hitl = HITLManager(calendar_adapter=calendar_adapter)
        action_id = await hitl.stage("schedule_event", {}, "Test")
        await hitl.confirm(action_id, "undo")
        action = await _get_action(action_id)
        assert action.status == "CANCELLED"

    async def test_undo_does_not_execute(self, calendar_adapter):
        """'undo' must NOT create any calendar events."""
        hitl = HITLManager(calendar_adapter=calendar_adapter)
        action_id = await hitl.stage(
            "schedule_event",
            {"title": "Ghost Event",
             "start": SLOT_START.isoformat(),
             "end":   SLOT_END.isoformat()},
            "Test",
        )
        await hitl.confirm(action_id, "undo")
        # No event written
        conflict = await calendar_adapter.check_overlap(SLOT_START, SLOT_END)
        assert conflict is None

    async def test_confirm_raises_for_unknown_id(self, calendar_adapter):
        hitl = HITLManager(calendar_adapter=calendar_adapter)
        with pytest.raises(ValueError, match="not found"):
            await hitl.confirm("nonexistent-id", "confirm")

    async def test_confirm_raises_if_already_confirmed(self, calendar_adapter):
        hitl = HITLManager(calendar_adapter=calendar_adapter)
        action = await _create_action(status="CONFIRMED")
        with pytest.raises(ValueError, match="already in state"):
            await hitl.confirm(action.id, "confirm")

    async def test_confirm_raises_if_already_cancelled(self, calendar_adapter):
        hitl = HITLManager(calendar_adapter=calendar_adapter)
        action = await _create_action(status="CANCELLED")
        with pytest.raises(ValueError, match="already in state"):
            await hitl.confirm(action.id, "confirm")

    async def test_confirm_raises_and_marks_expired_for_past_action(self, calendar_adapter):
        """Action past expires_at → status = EXPIRED, ValueError raised."""
        hitl = HITLManager(calendar_adapter=calendar_adapter)
        # Create action already expired (expires_at 1 hour ago)
        action = await _create_action(expires_offset_minutes=-60)
        with pytest.raises(ValueError, match="expired"):
            await hitl.confirm(action.id, "confirm")
        updated = await _get_action(action.id)
        assert updated.status == "EXPIRED"

    async def test_confirm_executes_schedule_event(self, calendar_adapter):
        """confirm('confirm') for schedule_event → event created in DB."""
        hitl = HITLManager(calendar_adapter=calendar_adapter)
        action_id = await hitl.stage(
            "schedule_event",
            {"title": "Contest Final",
             "start":  SLOT_START.isoformat(),
             "end":    SLOT_END.isoformat(),
             "source": "contest",
             "priority": 1},
            "Schedule Contest Final",
        )
        msg = await hitl.confirm(action_id, "confirm")
        assert "Contest Final" in msg

        conflict = await calendar_adapter.check_overlap(SLOT_START, SLOT_END)
        assert conflict is not None
        assert "Contest Final" in conflict.conflicting_title

    async def test_confirm_executes_delete_email(self, calendar_adapter):
        """confirm('confirm') for delete_email → email_adapter.delete() called."""
        stub = StubEmailAdapter()
        hitl = HITLManager(calendar_adapter=calendar_adapter, email_adapter=stub)
        action_id = await hitl.stage(
            "delete_email",
            {"email_id": "stub-email-002", "subject": "Spam"},
            "Delete spam email",
        )
        await hitl.confirm(action_id, "confirm")
        assert "stub-email-002" in stub.deleted


# ===========================================================================
# expire_stale()
# ===========================================================================


class TestHITLExpireStale:

    async def test_expire_stale_marks_past_actions_expired(self, calendar_adapter):
        hitl = HITLManager(calendar_adapter=calendar_adapter)

        # 2 past actions (expired)
        past_1 = await _create_action(expires_offset_minutes=-60)
        past_2 = await _create_action(expires_offset_minutes=-1)

        # 1 future action (should NOT be expired)
        future = await _create_action(expires_offset_minutes=60)

        count = await hitl.expire_stale()

        assert count == 2
        assert (await _get_action(past_1.id)).status == "EXPIRED"
        assert (await _get_action(past_2.id)).status == "EXPIRED"
        assert (await _get_action(future.id)).status == "REQUIRES_REVIEW"

    async def test_expire_stale_returns_zero_when_nothing_stale(self, calendar_adapter):
        hitl = HITLManager(calendar_adapter=calendar_adapter)
        await _create_action(expires_offset_minutes=120)  # future
        count = await hitl.expire_stale()
        assert count == 0

    async def test_expire_stale_ignores_confirmed_actions(self, calendar_adapter):
        """Already-resolved (CONFIRMED/CANCELLED) actions are not re-expired."""
        hitl = HITLManager(calendar_adapter=calendar_adapter)
        # In the past but already CONFIRMED
        already_confirmed = await _create_action(status="CONFIRMED", expires_offset_minutes=-60)
        count = await hitl.expire_stale()
        assert count == 0
        assert (await _get_action(already_confirmed.id)).status == "CONFIRMED"
