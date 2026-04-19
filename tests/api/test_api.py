"""
tests/api/test_api.py — API-level tests for src/api/main.py.

Test strategy:
  - Override app.router.lifespan_context with a test-specific context manager.
    This replaces the production lifespan (which needs GOOGLE_API_KEY, real
    contest providers, etc.) with one that injects pre-built test adapters.
  - All adapters are injected via the test lifespan into app.state.
  - The LLM is mocked to return deterministic routing decisions.
  - After each test, app.state is cleared to prevent state leaking between tests.

Exit criteria (Phase 4, Design Doc §10):
  ✓ GET  /health → 200 {"status": "ok"}
  ✓ POST /agent/run → 200 with response, agent_used, pending_action_id
  ✓ POST /agent/run with low-confidence → "CLARIFY" response
  ✓ POST /agent/run with blank input → 422 (validation error)
  ✓ POST /agent/confirm/{id} confirm → 200 CONFIRMED
  ✓ POST /agent/confirm/{id} undo → 200 CANCELLED
  ✓ POST /agent/confirm/{id} not found → 404
  ✓ POST /agent/confirm/{id} already resolved → 409
  ✓ POST /agent/confirm/{id} expired → 410
  ✓ GET  /agent/pending empty → {"pending": [], "count": 0}
  ✓ GET  /agent/pending with staged items → count > 0
  ✓ GET  /agent/pending only REQUIRES_REVIEW shown (not CONFIRMED)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.hitl import HITLManager
from src.parsers import RoutingDecision
from src.tools.calendar import SQLiteCalendarAdapter
from src.tools.email import StubEmailAdapter

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

SLOT_START = datetime(2026, 8, 1, 14, 0, tzinfo=timezone.utc)
SLOT_END   = datetime(2026, 8, 1, 16, 0, tzinfo=timezone.utc)


def _make_routing_llm(agent: str = "CONTEST_AGENT", confidence: float = 0.95) -> MagicMock:
    """Build a mock LLM that returns a fixed RoutingDecision."""
    mock_chain = AsyncMock()
    mock_chain.ainvoke = AsyncMock(
        return_value=RoutingDecision(agent=agent, confidence=confidence)
    )
    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_chain)
    return mock_llm


def _make_contest_agent_mock(
    result: str = "Scheduled 'CF Round 987'.",
    pending_id: str | None = None,
) -> MagicMock:
    """Mock ContestAgent whose run() returns a fixed result dict."""
    mock = MagicMock()
    mock.run = AsyncMock(return_value={"result": result, "pending_action_id": pending_id})
    return mock


def _make_email_agent_mock() -> MagicMock:
    """Mock EmailAgent with no-op run()."""
    mock = MagicMock()
    mock.run = AsyncMock(return_value={"action_report": []})
    return mock


@pytest_asyncio.fixture
async def api_bundle(calendar_adapter, stub_email):
    """
    Yields (AsyncClient, HITLManager) with test adapters injected directly
    into app.state.

    Why not override the lifespan?
    httpx.ASGITransport does NOT send ASGI lifespan events — the lifespan
    context never runs. Setting app.state directly before the AsyncClient
    context is the correct pattern for ASGI unit tests.

    Teardown clears app.state so no attributes leak between tests.
    """
    from src.api.main import app

    hitl = HITLManager(calendar_adapter=calendar_adapter, email_adapter=stub_email)
    mock_llm      = _make_routing_llm("CONTEST_AGENT", 0.95)
    contest_agent = _make_contest_agent_mock()
    email_agent   = _make_email_agent_mock()

    # Directly inject test state — no lifespan needed with ASGITransport
    app.state.hitl_manager     = hitl
    app.state.calendar_adapter = calendar_adapter
    app.state.email_adapter    = stub_email
    app.state.contest_agent    = contest_agent
    app.state.email_agent      = email_agent
    app.state.llm              = mock_llm

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client, hitl

    # Prevent state leaking into the next test
    app.state._state.clear()



# ===========================================================================
# Health
# ===========================================================================


class TestHealth:

    async def test_health_returns_ok(self, api_bundle):
        client, _ = api_bundle
        response = await client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "version" in body


# ===========================================================================
# POST /agent/run
# ===========================================================================


class TestAgentRun:

    async def test_run_dispatches_contest_agent(self, api_bundle):
        """Happy path: high-confidence CONTEST_AGENT routing → 200 with result."""
        client, _ = api_bundle
        response = await client.post(
            "/agent/run",
            json={"user_input": "Find me a Codeforces contest this weekend"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["agent_used"] == "CONTEST_AGENT"
        assert "CF Round 987" in body["response"]
        assert body["pending_action_id"] is None

    async def test_run_returns_pending_action_id_when_hitl_staged(self, api_bundle):
        """
        When the contest agent stages a HITL action, pending_action_id flows
        into the response so the client knows to call /agent/confirm/{id}.
        """
        client, _ = api_bundle
        from src.api.main import app

        # Override contest agent to return a pending action id
        fake_id = str(uuid4())
        app.state.contest_agent = _make_contest_agent_mock(
            result="Found a 5-hour contest. Please confirm.",
            pending_id=fake_id,
        )

        response = await client.post(
            "/agent/run",
            json={"user_input": "Schedule a 5h contest"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["pending_action_id"] == fake_id
        assert "confirm" in body["response"].lower()

    async def test_run_clarify_when_low_confidence(self, api_bundle):
        """
        Confidence < 0.6 → supervisor returns CLARIFY, no agent dispatched.
        (Design Doc §8.1, confidence gate)
        """
        client, _ = api_bundle
        from src.api.main import app

        # Override LLM to return low confidence
        app.state.llm = _make_routing_llm("CONTEST_AGENT", confidence=0.3)

        response = await client.post(
            "/agent/run",
            json={"user_input": "maybe something idk"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["agent_used"] == "CLARIFY"
        assert body["response"]  # non-empty clarification message

    async def test_run_email_agent_routing(self, api_bundle):
        """EMAIL_AGENT routing path → email_agent.run() called, summary returned."""
        client, _ = api_bundle
        from src.api.main import app

        app.state.llm = _make_routing_llm("EMAIL_AGENT", 0.9)
        app.state.email_agent = _make_email_agent_mock()

        response = await client.post(
            "/agent/run",
            json={"user_input": "Triage my inbox"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["agent_used"] == "EMAIL_AGENT"

    async def test_run_empty_input_rejected(self, api_bundle):
        """Pydantic min_length=1 on user_input → 422 Unprocessable Entity."""
        client, _ = api_bundle
        response = await client.post("/agent/run", json={"user_input": ""})
        assert response.status_code == 422

    async def test_run_missing_input_rejected(self, api_bundle):
        """Missing user_input field → 422 Unprocessable Entity."""
        client, _ = api_bundle
        response = await client.post("/agent/run", json={})
        assert response.status_code == 422

    async def test_run_response_schema_complete(self, api_bundle):
        """Response always has all 3 RunResponse fields."""
        client, _ = api_bundle
        response = await client.post(
            "/agent/run", json={"user_input": "find a contest"}
        )
        body = response.json()
        assert "response" in body
        assert "agent_used" in body
        assert "pending_action_id" in body


# ===========================================================================
# POST /agent/confirm/{action_id}
# ===========================================================================


class TestAgentConfirm:

    async def _stage(self, hitl: HITLManager, *, expires_offset_minutes: int = 30) -> str:
        """Helper: stage a schedule_event action and return its ID."""
        return await hitl.stage(
            action_type="schedule_event",
            payload={
                "title": "Test Event",
                "start": SLOT_START.isoformat(),
                "end":   SLOT_END.isoformat(),
                "source": "contest",
            },
            description="Test pending action",
        )

    async def test_confirm_action_returns_confirmed(self, api_bundle):
        """confirm → 200 with status CONFIRMED."""
        client, hitl = api_bundle
        action_id = await self._stage(hitl)

        response = await client.post(
            f"/agent/confirm/{action_id}",
            json={"decision": "confirm"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "CONFIRMED"
        assert body["action_id"] == action_id
        assert body["message"]   # non-empty

    async def test_undo_action_returns_cancelled(self, api_bundle):
        """undo → 200 with status CANCELLED, no execution."""
        client, hitl = api_bundle
        action_id = await self._stage(hitl)

        response = await client.post(
            f"/agent/confirm/{action_id}",
            json={"decision": "undo"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "CANCELLED"

    async def test_confirm_not_found_returns_404(self, api_bundle):
        """Non-existent action_id → 404."""
        client, _ = api_bundle
        response = await client.post(
            "/agent/confirm/nonexistent-uuid-12345",
            json={"decision": "confirm"},
        )
        assert response.status_code == 404

    async def test_confirm_already_resolved_returns_409(self, api_bundle):
        """Confirming an already-confirmed action → 409 Conflict."""
        client, hitl = api_bundle
        action_id = await self._stage(hitl)

        # First confirm (succeeds)
        res1 = await client.post(f"/agent/confirm/{action_id}", json={"decision": "undo"})
        assert res1.status_code == 200

        # Second confirm on the same action → 409
        res2 = await client.post(f"/agent/confirm/{action_id}", json={"decision": "confirm"})
        assert res2.status_code == 409

    async def test_confirm_expired_returns_410(self, api_bundle):
        """Expired action → 410 Gone."""
        client, hitl = api_bundle
        # Manually insert an already-expired action
        from src.db import PendingAction, get_async_session
        expired = PendingAction(
            id=str(uuid4()),
            action_type="schedule_event",
            payload=json.dumps({"title": "Ghost Event"}),
            description="Expired",
            status="REQUIRES_REVIEW",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        async with get_async_session() as session:
            session.add(expired)
            await session.commit()

        response = await client.post(
            f"/agent/confirm/{expired.id}",
            json={"decision": "confirm"},
        )
        assert response.status_code == 410

    async def test_confirm_invalid_decision_rejected(self, api_bundle):
        """decision must be 'confirm' or 'undo' — anything else → 422."""
        client, hitl = api_bundle
        action_id = await self._stage(hitl)
        response = await client.post(
            f"/agent/confirm/{action_id}",
            json={"decision": "delete_please"},
        )
        assert response.status_code == 422


# ===========================================================================
# GET /agent/pending
# ===========================================================================


class TestAgentPending:

    async def test_list_pending_empty(self, api_bundle):
        """Empty DB → {"pending": [], "count": 0}."""
        client, _ = api_bundle
        response = await client.get("/agent/pending")
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 0
        assert body["pending"] == []

    async def test_list_pending_shows_staged_actions(self, api_bundle):
        """Staged actions appear in the list."""
        client, hitl = api_bundle
        await hitl.stage("schedule_event", {"title": "CF Round 987"}, "Schedule CF Round 987")
        await hitl.stage("delete_email",   {"email_id": "spam-001"},  "Delete spam email")

        response = await client.get("/agent/pending")
        body = response.json()
        assert body["count"] == 2
        types = {item["action_type"] for item in body["pending"]}
        assert "schedule_event" in types
        assert "delete_email"   in types

    async def test_list_pending_only_requires_review(self, api_bundle):
        """
        CONFIRMED and CANCELLED actions must NOT appear in the list.
        Only REQUIRES_REVIEW rows are returned.
        """
        client, hitl = api_bundle

        # Stage 3, confirm or cancel 2 of them
        id1 = await hitl.stage("schedule_event", {"title": "A"}, "A")
        id2 = await hitl.stage("schedule_event", {"title": "B"}, "B")
        id3 = await hitl.stage("schedule_event",
                               {"title": "C",
                                "start": SLOT_START.isoformat(),
                                "end":   SLOT_END.isoformat(),
                                "source": "contest"},
                               "C")

        await hitl.confirm(id1, "undo")     # → CANCELLED
        await hitl.confirm(id3, "confirm")  # → CONFIRMED (creates event)

        response = await client.get("/agent/pending")
        body = response.json()
        assert body["count"] == 1
        assert body["pending"][0]["id"] == id2

    async def test_list_pending_item_has_required_fields(self, api_bundle):
        """Each item has all PendingActionItem fields."""
        client, hitl = api_bundle
        await hitl.stage("schedule_event", {"title": "T"}, "Test")

        response = await client.get("/agent/pending")
        item = response.json()["pending"][0]
        for field in ("id", "action_type", "description", "status", "expires_at", "created_at"):
            assert field in item, f"Missing field: {field}"

    async def test_list_pending_ordered_by_created_at(self, api_bundle):
        """Items are returned in creation order (oldest first)."""
        client, hitl = api_bundle
        id1 = await hitl.stage("schedule_event", {"title": "First"},  "First")
        id2 = await hitl.stage("schedule_event", {"title": "Second"}, "Second")
        id3 = await hitl.stage("delete_email",   {"email_id": "x"},   "Third")

        response = await client.get("/agent/pending")
        ids = [item["id"] for item in response.json()["pending"]]
        assert ids == [id1, id2, id3]
