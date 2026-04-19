"""
tests/api/test_observability.py — Phase 5 observability tests.

Covers:
  - GET /metrics counts (empty DB, with pending, with events)
  - Request latency middleware (X-Process-Time-Ms header)
  - AgentLog DB write after a successful /agent/run call
  - expire_stale_loop background task (cancels cleanly, actually calls expire_stale)
  - configure_logging() smoke test

Test isolation strategy (same as test_api.py):
  - ASGITransport does NOT run the lifespan.
  - app.state is set directly in the obs_client fixture.
  - test_db (autouse) patches src.db.get_async_session → test DB.
  - AgentLog / metrics queries therefore hit the same isolated in-memory DB.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from src.db import AgentLog, Event, PendingAction, get_async_session
from src.hitl import HITLManager
from src.parsers import RoutingDecision

SLOT_START = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
SLOT_END   = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Shared fixture — minimal state needed for observability tests
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def obs_client(calendar_adapter, stub_email):
    """
    Lightweight test client for observability tests.

    Injects test adapters directly into app.state (ASGITransport skips lifespan).
    The mock LLM is wired to route to CONTEST_AGENT so /agent/run succeeds.
    """
    from src.api.main import app

    hitl = HITLManager(calendar_adapter=calendar_adapter, email_adapter=stub_email)

    # Mock LLM → CONTEST_AGENT
    mock_chain = AsyncMock()
    mock_chain.ainvoke = AsyncMock(
        return_value=RoutingDecision(agent="CONTEST_AGENT", confidence=0.92)
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_chain)

    # Mock contest agent → returns success immediately
    mock_contest = MagicMock()
    mock_contest.run = AsyncMock(
        return_value={"result": "Scheduled!", "pending_action_id": None}
    )

    app.state.hitl_manager     = hitl
    app.state.calendar_adapter = calendar_adapter
    app.state.email_adapter    = stub_email
    app.state.contest_agent    = mock_contest
    app.state.email_agent      = MagicMock()
    app.state.llm              = mock_llm

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client, hitl

    app.state._state.clear()


# ===========================================================================
# GET /metrics
# ===========================================================================


class TestMetrics:

    async def test_metrics_empty_db(self, obs_client):
        """All counters are 0 when the database is empty."""
        client, _ = obs_client
        response = await client.get("/metrics")
        assert response.status_code == 200
        body = response.json()
        assert body["pending_requires_review"] == 0
        assert body["pending_expired"]         == 0
        assert body["events_total"]            == 0

    async def test_metrics_counts_pending_requires_review(self, obs_client):
        """Staged actions appear in pending_requires_review count."""
        client, hitl = obs_client
        await hitl.stage("schedule_event", {"title": "A"}, "Action A")
        await hitl.stage("delete_email",   {"email_id": "x"}, "Action B")

        response = await client.get("/metrics")
        assert response.json()["pending_requires_review"] == 2

    async def test_metrics_counts_pending_expired(self, obs_client):
        """Actions marked EXPIRED appear in pending_expired count."""
        client, _ = obs_client
        # Insert an already-expired action directly
        import json
        expired = PendingAction(
            id=str(uuid4()),
            action_type="schedule_event",
            payload=json.dumps({"title": "Ghost"}),
            description="Already expired",
            status="EXPIRED",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        async with get_async_session() as session:
            session.add(expired)
            await session.commit()

        response = await client.get("/metrics")
        assert response.json()["pending_expired"] == 1

    async def test_metrics_counts_events_total(self, obs_client, calendar_adapter):
        """Events created via the calendar adapter count in events_total."""
        client, _ = obs_client
        await calendar_adapter.create_event("E1", SLOT_START, SLOT_END, "contest")
        await calendar_adapter.create_event(
            "E2",
            SLOT_START + timedelta(hours=3),
            SLOT_END   + timedelta(hours=3),
            "meeting",
        )

        response = await client.get("/metrics")
        assert response.json()["events_total"] == 2

    async def test_metrics_response_has_all_fields(self, obs_client):
        """MetricsResponse always contains all three required fields."""
        client, _ = obs_client
        body = (await client.get("/metrics")).json()
        for field in ("pending_requires_review", "pending_expired", "events_total"):
            assert field in body, f"Missing field: {field}"


# ===========================================================================
# Middleware — X-Process-Time-Ms header
# ===========================================================================


class TestMiddleware:

    async def test_health_response_has_process_time_header(self, obs_client):
        """Every response carries X-Process-Time-Ms added by the middleware."""
        client, _ = obs_client
        response = await client.get("/health")
        # httpx lowercases header names
        assert "x-process-time-ms" in response.headers

    async def test_process_time_header_is_numeric(self, obs_client):
        """X-Process-Time-Ms value is a non-negative integer string."""
        client, _ = obs_client
        response = await client.get("/health")
        header_val = response.headers["x-process-time-ms"]
        assert header_val.isdigit(), f"Expected digits, got: {header_val!r}"

    async def test_agent_run_response_has_process_time_header(self, obs_client):
        """/agent/run responses also carry the latency header."""
        client, _ = obs_client
        response = await client.post("/agent/run", json={"user_input": "find a contest"})
        assert response.status_code == 200
        assert "x-process-time-ms" in response.headers


# ===========================================================================
# AgentLog write (Phase 5)
# ===========================================================================


class TestAgentLog:

    async def test_agent_log_written_after_successful_run(self, obs_client):
        """
        A successful POST /agent/run writes exactly one AgentLog row to the DB.
        """
        client, _ = obs_client
        response = await client.post(
            "/agent/run",
            json={"user_input": "find me a Codeforces contest"},
        )
        assert response.status_code == 200

        async with get_async_session() as session:
            result = await session.execute(select(AgentLog))
            logs = result.scalars().all()

        assert len(logs) == 1
        log_row = logs[0]
        assert log_row.agent == "CONTEST_AGENT"
        assert "find me a Codeforces contest" in log_row.user_input
        assert log_row.duration_ms >= 0
        assert log_row.session_id  # non-empty UUID

    async def test_agent_log_not_written_on_validation_error(self, obs_client):
        """
        A 422 (empty input rejected by Pydantic) must not write any AgentLog row.
        """
        client, _ = obs_client
        response = await client.post("/agent/run", json={"user_input": ""})
        assert response.status_code == 422

        async with get_async_session() as session:
            result = await session.execute(select(AgentLog))
            logs = result.scalars().all()

        assert len(logs) == 0

    async def test_multiple_runs_write_separate_log_rows(self, obs_client):
        """Each /agent/run call produces its own AgentLog row with a unique session_id."""
        client, _ = obs_client
        await client.post("/agent/run", json={"user_input": "find contest A"})
        await client.post("/agent/run", json={"user_input": "find contest B"})

        async with get_async_session() as session:
            result = await session.execute(select(AgentLog))
            logs = result.scalars().all()

        assert len(logs) == 2
        session_ids = {row.session_id for row in logs}
        assert len(session_ids) == 2, "Each run should have a unique session_id"


# ===========================================================================
# Background task — expire_stale_loop
# ===========================================================================


class TestExpireStaleLoop:

    async def test_expire_stale_loop_calls_expire_stale(self):
        """
        The loop calls hitl_manager.expire_stale() at least once during its run.
        Uses a very short interval (10ms) and a 100ms wait to guarantee ≥1 tick.
        """
        from src.tasks import expire_stale_loop

        mock_hitl = MagicMock()
        mock_hitl.expire_stale = AsyncMock(return_value=0)

        task = asyncio.create_task(
            expire_stale_loop(mock_hitl, interval_seconds=0.01)
        )
        await asyncio.sleep(0.10)  # 100ms wait → ~10 ticks at 10ms interval
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert mock_hitl.expire_stale.call_count >= 1

    async def test_expire_stale_loop_cancels_cleanly(self):
        """
        Cancelling the loop does not raise an unexpected exception —
        only asyncio.CancelledError is propagated, which we suppress.
        """
        from src.tasks import expire_stale_loop

        mock_hitl = MagicMock()
        mock_hitl.expire_stale = AsyncMock(return_value=0)

        task = asyncio.create_task(
            expire_stale_loop(mock_hitl, interval_seconds=60)  # long interval
        )
        await asyncio.sleep(0)  # yield to let the task start
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # expected — loop re-raises it

        assert task.done()
        assert task.cancelled()


# ===========================================================================
# configure_logging smoke test
# ===========================================================================


class TestConfigureLogging:

    def test_configure_logging_runs_without_error(self):
        """configure_logging() must not raise for any valid log level."""
        from src.logger import configure_logging

        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            configure_logging(log_level=level)  # must not raise
