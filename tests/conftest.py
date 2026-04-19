"""
tests/conftest.py — Shared pytest fixtures for all test suites.

Key decisions:
  - test_db (autouse=True): Each test gets a fresh in-memory SQLite database.
    Patches src.db.async_engine and src.db._async_session_factory for the
    duration of the test, then restores the originals.

  - calendar_adapter: Returns a fresh SQLiteCalendarAdapter pointing at the
    test DB (which test_db has already set up).

  - stub_email: Returns a fresh StubEmailAdapter with no prior state.

Design note on asyncio.Lock:
  CalendarAdapter._lock is a class-level Lock. Python 3.10+ asyncio Locks
  do not bind to a specific event loop at creation time; they bind at first
  acquire(). This means the shared class-level lock works correctly across
  test functions with pytest-asyncio's per-function event loop scoping.
"""
from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel

from src.tools.calendar import CalendarAdapter, SQLiteCalendarAdapter
from src.tools.email import StubEmailAdapter

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(autouse=True)
async def test_db():
    """
    Provide a fresh in-memory SQLite database for every test.

    Patches the module-level async_engine and _async_session_factory in src.db
    so that get_async_session() — used by all adapters — connects to the
    in-memory DB instead of agent_memory.db.

    Teardown restores the original engine so tests are fully isolated.
    """
    import src.db as db_module

    # Create a fresh in-memory engine for this test.
    # StaticPool: all sessions within this test share ONE physical connection,
    # so committed data from one session is immediately visible to others.
    # Without StaticPool, sqlite:///:memory: gives each connection its own
    # isolated database, breaking inter-session visibility in concurrent tests.
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Patch module-level references used by get_async_session()
    original_engine  = db_module.async_engine
    original_factory = db_module._async_session_factory

    db_module.async_engine          = engine
    db_module._async_session_factory = session_factory

    # Create all tables in the in-memory DB
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    yield  # --- test runs here ---

    # Tear down: dispose engine, restore originals
    await engine.dispose()
    db_module.async_engine          = original_engine
    db_module._async_session_factory = original_factory


@pytest.fixture
def calendar_adapter() -> SQLiteCalendarAdapter:
    """Fresh SQLiteCalendarAdapter pointed at the test DB (set up by test_db)."""
    return SQLiteCalendarAdapter()


@pytest.fixture
def stub_email() -> StubEmailAdapter:
    """Fresh StubEmailAdapter with no prior replies or deletions."""
    adapter = StubEmailAdapter()
    return adapter


@pytest.fixture(autouse=True)
def reset_calendar_lock():
    """
    Reset CalendarAdapter._lock before every test.

    The class-level asyncio.Lock is shared across all instances. Because
    pytest-asyncio creates a NEW event loop per test function, a Lock that was
    in the "waiting" state under the OLD loop can leave stale asyncio.Future
    objects in _lock._waiters. Those stale Futures corrupt the lock's behaviour
    in the new loop, causing concurrent acquires to never complete.

    Creating a new Lock() at the start of each test guarantees a clean slate.
    """
    CalendarAdapter._lock = asyncio.Lock()
    yield
    # Post-test reset ensures no stale state bleeds into the next test.
    CalendarAdapter._lock = asyncio.Lock()
