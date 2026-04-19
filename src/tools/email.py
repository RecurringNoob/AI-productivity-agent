"""
src/tools/email.py — EmailAdapter abstract base class and StubEmailAdapter.

Design Document Reference: Section 9.3 (Email Adapter), Section 4 (Component Breakdown)

Adapters:
  StubEmailAdapter  — hardcoded 3-email inbox for development and unit tests.
                      Tracks replies and deletions in-memory for test assertions.
  GmailEmailAdapter — production adapter (stub only — swap in via config in Phase 4+).
"""
from __future__ import annotations

import structlog
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import List

from src.parsers import EmailDTO

log = structlog.get_logger(__name__)


# ==============================================================================
# Abstract Base Class
# ==============================================================================


class EmailAdapter(ABC):
    """
    Abstract interface for email backends.

    Concrete implementations: StubEmailAdapter (dev/test), GmailEmailAdapter (prod).
    Swap the implementation at startup via lifespan injection in src/api/main.py.
    """

    @abstractmethod
    async def get_unread(self) -> List[EmailDTO]:
        """Fetch all unread emails from the mailbox."""
        ...

    @abstractmethod
    async def reply(self, email_id: str, body: str) -> None:
        """Reply to the email identified by email_id."""
        ...

    @abstractmethod
    async def delete(self, email_id: str) -> None:
        """
        Delete the email identified by email_id.

        NOTE: Direct deletion is NEVER called by agents.
        Agents always stage a PendingAction via HITLManager.stage().
        This method is executed only after the user confirms the action
        via POST /agent/confirm/{id}. (Design Doc §5.2, Improvement 3)
        """
        ...


# ==============================================================================
# Stub Implementation — Development & Tests
# ==============================================================================


class StubEmailAdapter(EmailAdapter):
    """
    Development and test email adapter.

    Returns a hardcoded 3-email inbox:
      1. MEETING_REQUEST  — Alice requesting a meeting tomorrow at 2pm
      2. SPAM             — Promotional newsletter
      3. GENERAL          — Bob checking on project status

    Call reset() between tests to clear recorded replies and deletions.
    """

    def __init__(self) -> None:
        # In-memory logs — inspectable in test assertions
        self.replies: list[dict] = []
        self.deleted: list[str] = []

    def reset(self) -> None:
        """Clear all recorded interactions. Call between tests."""
        self.replies.clear()
        self.deleted.clear()

    async def get_unread(self) -> List[EmailDTO]:
        now = datetime.now(timezone.utc)
        return [
            EmailDTO(
                id="stub-email-001",
                sender="alice@example.com",
                subject="Quick Sync Tomorrow",
                body=(
                    "Hi, can we meet tomorrow at 2pm for a project sync? "
                    "Shouldn't take more than an hour."
                ),
                received_at=now - timedelta(minutes=10),
            ),
            EmailDTO(
                id="stub-email-002",
                sender="newsletter@promotions.example.com",
                subject="🔥 Limited Time Offer — 80% OFF Today Only!",
                body=(
                    "Congratulations! You have been selected for our exclusive deal. "
                    "Click here to claim your prize NOW."
                ),
                received_at=now - timedelta(minutes=5),
            ),
            EmailDTO(
                id="stub-email-003",
                sender="bob@example.com",
                subject="Project status check",
                body="Hey, just wanted to check in on where we stand with the project. No rush.",
                received_at=now,
            ),
        ]

    async def reply(self, email_id: str, body: str) -> None:
        log.info("stub_email_reply", email_id=email_id, body_preview=body[:80])
        self.replies.append({"email_id": email_id, "body": body})

    async def delete(self, email_id: str) -> None:
        log.info("stub_email_delete", email_id=email_id)
        self.deleted.append(email_id)


# ==============================================================================
# Gmail Implementation — Production (Phase 4+ swap-in)
# ==============================================================================


class GmailEmailAdapter(EmailAdapter):
    """
    Production Gmail adapter.

    Not yet implemented — swap in during Phase 4 by setting GMAIL_CREDENTIALS_PATH.
    Replace StubEmailAdapter with this class in src/api/main.py lifespan.
    """

    def __init__(self, credentials_path: str) -> None:
        self.credentials_path = credentials_path
        log.warning(
            "gmail_adapter_not_implemented",
            note="GmailEmailAdapter is a stub. Set credentials_path and implement OAuth flow.",
        )

    async def get_unread(self) -> List[EmailDTO]:
        raise NotImplementedError("GmailEmailAdapter.get_unread() not yet implemented.")

    async def reply(self, email_id: str, body: str) -> None:
        raise NotImplementedError("GmailEmailAdapter.reply() not yet implemented.")

    async def delete(self, email_id: str) -> None:
        raise NotImplementedError("GmailEmailAdapter.delete() not yet implemented.")
