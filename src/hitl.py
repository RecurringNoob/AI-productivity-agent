"""
src/hitl.py — Human-in-the-Loop (HITL) Manager.

Design Document Reference: Section 9.5 (HITL Manager), Improvement 3

Responsibilities:
  stage()   — Persist a pending action to DB, return its UUID for the API response.
  confirm() — Mark CONFIRMED or CANCELLED; if confirmed, execute the action.
  expire_stale() — Bulk-mark REQUIRES_REVIEW rows past expires_at as EXPIRED
                   (called on server startup and periodically via background task).

ARCHITECTURE RULE (Design Doc §5.2):
  Agents NEVER directly call calendar.create_event() or email.delete().
  They ALWAYS call hitl.stage() for high-stakes or irreversible operations.
  Execution happens only after the user confirms via POST /agent/confirm/{id}.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import structlog
from sqlalchemy import select, update

from src.db import PendingAction, get_async_session

if TYPE_CHECKING:
    from src.tools.calendar import CalendarAdapter
    from src.tools.email import EmailAdapter

log = structlog.get_logger(__name__)

HITL_EXPIRY_MINUTES: int = int(os.getenv("HITL_EXPIRY_MINUTES", "30"))


class HITLManager:
    """
    Manages the lifecycle of pending (human-confirmation-required) actions.

    Adapters are optional at construction time — inject them when the server
    starts up so that confirm() can execute the staged action.
    (Design Doc §9.5, Phase 4 wires adapters via lifespan injection)
    """

    def __init__(
        self,
        calendar_adapter: CalendarAdapter | None = None,
        email_adapter: EmailAdapter | None = None,
    ) -> None:
        self.calendar_adapter = calendar_adapter
        self.email_adapter    = email_adapter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def stage(
        self,
        action_type: str,
        payload: dict,
        description: str,
    ) -> str:
        """
        Persist a PendingAction and return its UUID.

        Called by agents for high-stakes or irreversible operations.
        The UUID is returned to the user so they can confirm/cancel via the API.

        Args:
            action_type:  "schedule_event" | "delete_email"
            payload:      JSON-serialisable dict of action parameters.
            description:  Human-readable summary shown to the user.

        Returns:
            UUID string (PendingAction.id)
        """
        action = PendingAction(
            id=str(uuid4()),
            action_type=action_type,
            payload=json.dumps(payload),
            description=description,
            status="REQUIRES_REVIEW",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=HITL_EXPIRY_MINUTES),
        )

        async with get_async_session() as session:
            session.add(action)
            await session.commit()

        log.info(
            "hitl_staged",
            action_id=action.id,
            action_type=action_type,
            description=description[:80],
            expires_at=str(action.expires_at),
        )
        return action.id

    async def confirm(
        self,
        action_id: str,
        decision: Literal["confirm", "undo"],
    ) -> str:
        """
        Mark a pending action as CONFIRMED or CANCELLED.

        If confirmed, executes the staged action via the appropriate adapter.
        If cancelled, marks CANCELLED and returns immediately.

        Raises:
            ValueError if action not found or not in REQUIRES_REVIEW state.
        """
        async with get_async_session() as session:
            result = await session.execute(
                select(PendingAction).where(PendingAction.id == action_id)
            )
            action = result.scalars().first()

            if action is None:
                raise ValueError(f"PendingAction {action_id!r} not found.")

            if action.status != "REQUIRES_REVIEW":
                raise ValueError(
                    f"PendingAction {action_id!r} is already in state '{action.status}'."
                )

            # Check expiry
            now = datetime.now(timezone.utc)
            action_expires = (
                action.expires_at.replace(tzinfo=timezone.utc)
                if action.expires_at.tzinfo is None
                else action.expires_at
            )
            if now > action_expires:
                action.status = "EXPIRED"
                session.add(action)
                await session.commit()
                raise ValueError(f"PendingAction {action_id!r} has expired.")

            if decision == "confirm":
                outcome = await self._execute(action)
                action.status = "CONFIRMED"
            else:
                outcome = f"Action '{action.action_type}' cancelled by user."
                action.status = "CANCELLED"

            session.add(action)
            await session.commit()

        log.info(
            "hitl_resolved",
            action_id=action_id,
            decision=decision,
            status=action.status,
        )
        return outcome

    async def expire_stale(self) -> int:
        """
        Mark all REQUIRES_REVIEW actions past their expires_at as EXPIRED.

        Called on server startup and periodically via background task (Phase 5).
        Returns the number of actions expired.
        """
        now = datetime.now(timezone.utc)

        async with get_async_session() as session:
            result = await session.execute(
                select(PendingAction).where(
                    PendingAction.status == "REQUIRES_REVIEW",
                )
            )
            stale = [
                a for a in result.scalars().all()
                if (
                    a.expires_at.replace(tzinfo=timezone.utc)
                    if a.expires_at.tzinfo is None
                    else a.expires_at
                ) < now
            ]

            for action in stale:
                action.status = "EXPIRED"
                session.add(action)

            await session.commit()

        count = len(stale)
        if count:
            log.info("hitl_expire_stale", expired_count=count)
        return count

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    async def _execute(self, action: PendingAction) -> str:
        """
        Execute the confirmed action using the appropriate adapter.
        Adapters must be injected at construction time for this to work.
        """
        payload = json.loads(action.payload)

        if action.action_type == "schedule_event":
            if self.calendar_adapter is None:
                log.warning("hitl_execute_no_calendar", action_id=action.id)
                return "Executed (calendar adapter not configured)."

            start = datetime.fromisoformat(payload["start"])
            end   = datetime.fromisoformat(payload["end"])
            event = await self.calendar_adapter.create_event(
                title=payload["title"],
                start=start,
                end=end,
                source=payload.get("source", "contest"),
                priority=payload.get("priority", 1),
                external_id=payload.get("external_id"),
            )
            return f"Scheduled: '{event.title}' on {event.start_time.strftime('%Y-%m-%d %H:%M')} UTC."

        if action.action_type == "delete_email":
            if self.email_adapter is None:
                log.warning("hitl_execute_no_email", action_id=action.id)
                return "Executed (email adapter not configured)."

            await self.email_adapter.delete(payload["email_id"])
            return f"Deleted email: {payload.get('subject', payload['email_id'])}."

        log.warning("hitl_unknown_action_type", action_type=action.action_type)
        return f"Unknown action type '{action.action_type}'."
