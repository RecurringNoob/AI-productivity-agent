"""
src/db.py — SQLModel table definitions and async engine factory.

Single source of truth for all database schema and session management.
All four tables are defined here: Event, EmailAction, AgentLog, PendingAction.

Design Document Reference: Section 6.1 (Database Tables)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlmodel import Field, SQLModel

# ==============================================================================
# Configuration
# ==============================================================================

# Runtime URL uses the async aiosqlite driver.
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "sqlite+aiosqlite:///./agent_memory.db"
)

# Synchronous URL is exposed for Alembic migrations (alembic uses sync engine).
# migrations/env.py calls this directly.
SYNC_DATABASE_URL: str = DATABASE_URL.replace("sqlite+aiosqlite", "sqlite")

# ==============================================================================
# Async engine + session factory
# ==============================================================================

async_engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    connect_args={"check_same_thread": False},
)

_async_session_factory = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async DB session; auto-closed and rolled back on exception."""
    async with _async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """
    Create all tables via SQLModel metadata.

    Used in tests and development. Production deployments use `alembic upgrade head`.
    """
    async with async_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


# ==============================================================================
# Table: events
# Design Document §6.1 — stores calendar events from all sources
# ==============================================================================

class Event(SQLModel, table=True):
    id:          int | None = Field(default=None, primary_key=True)
    title:       str
    start_time:  datetime
    end_time:    datetime
    source:      str                              # "contest" | "meeting" | "manual"
    external_id: str | None = Field(default=None, unique=True)  # UNIQUE — prevents duplicate contest rows
    priority:    int = Field(default=0)           # 0 = normal, 1 = high (enables proactive reschedule suggestions)
    created_at:  datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table: email_actions
# Design Document §6.1 — audit log for all email triage decisions
# ==============================================================================

class EmailAction(SQLModel, table=True):
    id:         int | None = Field(default=None, primary_key=True)
    email_id:   str
    sender:     str
    subject:    str
    intent:     str   # MEETING_REQUEST | SPAM | GENERAL | NEEDS_CLARIFICATION
    action:     str   # scheduled | declined | deleted | logged | clarification_sent | pending_review
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table: agent_logs
# Design Document §6.1 — structured log of every agent interaction
# ==============================================================================

class AgentLog(SQLModel, table=True):
    id:          int | None = Field(default=None, primary_key=True)
    session_id:  str
    user_input:  str
    agent:       str
    outcome:     str
    duration_ms: int
    created_at:  datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ==============================================================================
# Table: pending_actions  (NEW in v3.0)
# Design Document §6.1 — stores HITL actions awaiting user confirmation
# ==============================================================================

class PendingAction(SQLModel, table=True):
    id:          str = Field(primary_key=True)         # UUID string
    action_type: str                                    # "schedule_event" | "delete_email" | ...
    payload:     str                                    # JSON-serialised action parameters
    description: str                                    # Human-readable summary for the UI
    status:      str = Field(default="REQUIRES_REVIEW") # → CONFIRMED | CANCELLED | EXPIRED
    expires_at:  datetime                               # Auto-cancel after N minutes if no response
    created_at:  datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
