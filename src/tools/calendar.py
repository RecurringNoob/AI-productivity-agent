"""
src/tools/calendar.py — CalendarAdapter ABC and SQLiteCalendarAdapter.

Design Document Reference: Section 9.1 (Calendar Adapter), Improvements 2 & 4

Key design decisions:
  _lock (class-level asyncio.Lock):
    Shared across ALL adapter instances and ALL subclasses. This is intentional.
    It ensures that concurrent API requests or fan-out email branches cannot both
    see a "free" slot before either write completes. (Improvement 2)

  safe_schedule (atomic wrapper):
    Acquires _lock before calling check_overlap + create_event.
    Callers (contest agent, email reduce node) MUST use safe_schedule rather than
    calling the two methods separately. (Design Doc §9.1)

  buffer_minutes (Improvement 4):
    Expands the overlap query window on both sides of the new event so that
    back-to-back events with no travel time are flagged as soft conflicts.
"""
from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Union

import structlog
from sqlalchemy import select

from src.db import Event, get_async_session
from src.parsers import ConflictInfo

log = structlog.get_logger(__name__)

DEFAULT_BUFFER_MINUTES: int = int(os.getenv("CALENDAR_TRAVEL_BUFFER_MINUTES", "15"))


# ==============================================================================
# Abstract Base Class
# ==============================================================================


class CalendarAdapter(ABC):
    """
    Abstract calendar backend.

    _lock is a CLASS-LEVEL asyncio.Lock — intentionally shared across all
    instances to serialise all check-then-write operations globally.
    (Design Doc §9.1, Improvement 2)
    """

    _lock: asyncio.Lock = asyncio.Lock()

    @abstractmethod
    async def check_overlap(
        self,
        start: datetime,
        end: datetime,
        buffer_minutes: int = 0,
    ) -> ConflictInfo | None:
        """
        Query for any existing event that overlaps the window
        [start - buffer_minutes, end + buffer_minutes].

        Returns ConflictInfo if a conflict is found, None if the slot is free.
        """
        ...

    @abstractmethod
    async def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        source: str,
        priority: int = 0,
        external_id: str | None = None,
    ) -> Event:
        """Persist a new calendar event and return it with its DB id populated."""
        ...

    async def safe_schedule(
        self,
        title: str,
        start: datetime,
        end: datetime,
        source: str,
        buffer_minutes: int = 0,
        priority: int = 0,
        external_id: str | None = None,
    ) -> Union[Event, ConflictInfo]:
        """
        Atomic check-then-write protected by the class-level lock.

        This is the ONLY way callers should schedule events. Calling
        check_overlap and create_event separately is not safe because
        concurrent coroutines could both observe a free slot before either
        write completes. (Design Doc §9.1, Improvement 2)

        Returns:
            Event        — if the slot was free and the event was created.
            ConflictInfo — if one or more existing events overlap the window
                           (including buffer expansion).
        """
        async with self._lock:
            conflict = await self.check_overlap(start, end, buffer_minutes)
            if conflict:
                log.info(
                    "safe_schedule_conflict",
                    new_title=title,
                    conflicting_title=conflict.conflicting_title,
                    buffer_applied=conflict.buffer_applied,
                )
                return conflict
            return await self.create_event(title, start, end, source, priority, external_id)


# ==============================================================================
# SQLite Implementation
# ==============================================================================


class SQLiteCalendarAdapter(CalendarAdapter):
    """
    Calendar adapter backed by the SQLite events table via async sessions.
    Used in development, tests, and (for now) production.
    """

    async def check_overlap(
        self,
        start: datetime,
        end: datetime,
        buffer_minutes: int = 0,
    ) -> ConflictInfo | None:
        """
        Range overlap query with optional soft-conflict buffer (Improvement 4).

        Buffered window:
            query_start = start - buffer_minutes
            query_end   = end   + buffer_minutes

        SQL overlap condition (Design Doc §9.1):
            event.start_time < query_end AND event.end_time > query_start

        This correctly identifies all events that overlap with the buffered window,
        including events that are adjacent (when buffer > 0).
        """
        query_start = start - timedelta(minutes=buffer_minutes)
        query_end   = end   + timedelta(minutes=buffer_minutes)

        async with get_async_session() as session:
            stmt = select(Event).where(
                Event.start_time < query_end,
                Event.end_time   > query_start,
            )
            result = await session.execute(stmt)
            existing = result.scalars().first()

        if existing:
            log.debug(
                "check_overlap_hit",
                existing_title=existing.title,
                buffer_minutes=buffer_minutes,
            )
            # SQLite stores datetimes as naive strings (no tz).
            # Reattach UTC so all returned datetimes are timezone-aware,
            # consistent with what was written in.
            start_aware = (
                existing.start_time.replace(tzinfo=timezone.utc)
                if existing.start_time.tzinfo is None
                else existing.start_time
            )
            end_aware = (
                existing.end_time.replace(tzinfo=timezone.utc)
                if existing.end_time.tzinfo is None
                else existing.end_time
            )
            return ConflictInfo(
                conflicting_title=existing.title,
                conflicting_start=start_aware,
                conflicting_end=end_aware,
                buffer_applied=buffer_minutes,
            )

        return None

    async def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        source: str,
        priority: int = 0,
        external_id: str | None = None,
    ) -> Event:
        """
        Persist a new Event to the database.

        expire_on_commit=False on the session factory means the returned
        Event object retains its attribute values after the session closes.
        """
        event = Event(
            title=title,
            start_time=start,
            end_time=end,
            source=source,
            priority=priority,
            external_id=external_id,
        )

        async with get_async_session() as session:
            session.add(event)
            await session.commit()
            await session.refresh(event)

        log.info(
            "calendar_write",
            title=title,
            start=str(start),
            end=str(end),
            source=source,
            priority=priority,
            external_id=external_id,
        )
        return event
