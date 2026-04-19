"""
src/api/main.py — FastAPI application with lifespan injection.

Design Document Reference: Section 10 (API Contract), Phase 5 (Observability)

Routes:
  GET  /health                       → liveness probe
  GET  /metrics                      → pending/expired/event counts (Phase 5)
  POST /agent/run                    → supervisor dispatch + AgentLog write
  POST /agent/confirm/{action_id}    → HITL confirm / cancel
  GET  /agent/pending                → list REQUIRES_REVIEW actions

Lifespan (Design Doc §10 / Phase 5):
  Startup:
    1. configure_logging()           — activate structured JSON logging
    2. create tables (dev/test)      — production uses `alembic upgrade head`
    3. build adapters + agents       — inject into app.state
    4. expire_stale() on startup     — clean up leftover pending actions
    5. asyncio.create_task(...)      — start background HITL expiry loop

  Shutdown:
    1. cancel expiry_task            — graceful asyncio.CancelledError
    2. dispose async engine          — close DB connection pool

Middleware:
  request_logger_middleware — measures latency per request, emits a structured
  log line, and sets X-Process-Time-Ms response header.

AgentLog (Phase 5):
  Written by /agent/run after every successful dispatch. Non-fatal: a DB write
  failure is logged as a warning but does not fail the response.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid as _uuid
from contextlib import asynccontextmanager
from datetime import timezone

import structlog
from fastapi import FastAPI, HTTPException, Request
from sqlalchemy import func, select

import src.db as db_module
from src.agents.contest import ContestAgent
from src.agents.email import EmailAgent
from src.api.schemas import (
    ConfirmRequest,
    ConfirmResponse,
    MetricsResponse,
    PendingActionItem,
    PendingListResponse,
    RunRequest,
    RunResponse,
)
from src.db import AgentLog, Event, PendingAction, get_async_session
from src.hitl import HITLManager
from src.llm_client import get_llm
from src.logger import configure_logging
from src.supervisor import dispatch
from src.tasks import expire_stale_loop
from src.tools.calendar import SQLiteCalendarAdapter
from src.tools.contests import AggregatingContestProvider, CodeforcesProvider, LeetCodeProvider
from src.tools.email import StubEmailAdapter

log = structlog.get_logger(__name__)

HITL_EXPIRY_CHECK_SECONDS: int = int(os.getenv("HITL_EXPIRY_CHECK_SECONDS", "300"))


# ==============================================================================
# Lifespan
# ==============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup / shutdown.

    All adapters, agents, and the LLM are built once and stored on app.state.
    Route handlers read from app.state — no singletons, no global mutable state.
    """
    # ── Startup ────────────────────────────────────────────────────────────────

    # 1. Activate structured JSON logging (Phase 5)
    configure_logging(log_level=os.getenv("LOG_LEVEL", "INFO"))
    log.info("startup_begin")

    # 2. Ensure tables exist (production: run `alembic upgrade head` before deploying)
    async with db_module.async_engine.begin() as conn:
        from sqlmodel import SQLModel
        await conn.run_sync(SQLModel.metadata.create_all)

    # 3. Build adapters
    calendar_adapter = SQLiteCalendarAdapter()
    email_adapter    = StubEmailAdapter()        # swap with GmailEmailAdapter in prod

    contest_provider = AggregatingContestProvider([
        CodeforcesProvider(),
        LeetCodeProvider(),
    ])

    hitl_manager = HITLManager(
        calendar_adapter=calendar_adapter,
        email_adapter=email_adapter,
    )

    # 4. Expire stale HITL actions from previous server runs (Design Doc §9.5)
    expired = await hitl_manager.expire_stale()
    if expired:
        log.info("startup_hitl_expired_stale", count=expired)

    # 5. Build LLM — non-fatal if GOOGLE_API_KEY is missing in dev environments
    try:
        llm = get_llm()
    except RuntimeError as exc:
        log.warning("startup_llm_unavailable", error=str(exc))
        llm = None

    # 6. Build agents
    contest_agent = ContestAgent(
        contest_provider=contest_provider,
        calendar_adapter=calendar_adapter,
        hitl_manager=hitl_manager,
        llm=llm,
    )
    email_agent = EmailAgent(
        email_adapter=email_adapter,
        calendar_adapter=calendar_adapter,
        hitl_manager=hitl_manager,
        llm=llm,
    )

    # 7. Inject into app.state
    app.state.calendar_adapter = calendar_adapter
    app.state.email_adapter    = email_adapter
    app.state.hitl_manager     = hitl_manager
    app.state.contest_agent    = contest_agent
    app.state.email_agent      = email_agent
    app.state.llm              = llm

    # 8. Start background HITL expiry task (Phase 5, Design Doc §9.5)
    app.state.expiry_task = asyncio.create_task(
        expire_stale_loop(hitl_manager, interval_seconds=HITL_EXPIRY_CHECK_SECONDS)
    )

    log.info("startup_complete")
    yield  # ← server is running here

    # ── Shutdown ───────────────────────────────────────────────────────────────
    log.info("shutdown_begin")

    # Cancel the background task cleanly
    if task := getattr(app.state, "expiry_task", None):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    await db_module.async_engine.dispose()
    log.info("shutdown_complete")


# ==============================================================================
# Application
# ==============================================================================


app = FastAPI(
    title="AI Productivity Agent v3.0",
    description=(
        "Hardened AI agent for contest scheduling and email triage. "
        "Implements Human-in-the-Loop (HITL) for high-stakes actions."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


# ==============================================================================
# Middleware — per-request latency logging (Phase 5)
# ==============================================================================


@app.middleware("http")
async def request_logger_middleware(request: Request, call_next):
    """
    Measure end-to-end latency for every HTTP request.

    Emits a structured log line:
        {"event": "request", "method": "POST", "path": "/agent/run",
         "status": 200, "duration_ms": 42}

    Sets X-Process-Time-Ms response header for client-side monitoring.
    """
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = int((time.perf_counter() - start) * 1000)

    log.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
    )
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    return response


# ==============================================================================
# Routes
# ==============================================================================


@app.get("/health", tags=["Ops"])
async def health():
    """Liveness probe — always returns 200 if the server is running."""
    return {"status": "ok", "version": "3.0.0"}


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    tags=["Ops"],
    summary="Lightweight operational metrics snapshot",
)
async def metrics_endpoint(_: Request) -> MetricsResponse:
    """
    Returns counts of:
      - pending_requires_review : HITL actions awaiting user confirmation
      - pending_expired         : HITL actions that timed out without a response
      - events_total            : total calendar events scheduled
    """
    async with get_async_session() as session:
        r1 = await session.execute(
            select(func.count(PendingAction.id))
            .where(PendingAction.status == "REQUIRES_REVIEW")
        )
        pending_requires_review: int = r1.scalar_one()

        r2 = await session.execute(
            select(func.count(PendingAction.id))
            .where(PendingAction.status == "EXPIRED")
        )
        pending_expired: int = r2.scalar_one()

        r3 = await session.execute(select(func.count(Event.id)))
        events_total: int = r3.scalar_one()

    return MetricsResponse(
        pending_requires_review=pending_requires_review,
        pending_expired=pending_expired,
        events_total=events_total,
    )


@app.post(
    "/agent/run",
    response_model=RunResponse,
    tags=["Agent"],
    summary="Dispatch a user request to the appropriate agent",
)
async def run_agent(body: RunRequest, request: Request) -> RunResponse:
    """
    Classify and dispatch the user's request via the Supervisor.

    On success writes one AgentLog row (Phase 5 observability). The DB write
    is non-fatal — a failure is logged as a warning but the response still
    returns normally.

    Routing:
      CONTEST_AGENT → contest scheduling
      EMAIL_AGENT   → inbox triage
      GENERAL       → answered by LLM directly
      CLARIFY       → returned when confidence < 0.6
    """
    session_id = str(_uuid.uuid4())
    t0 = time.perf_counter()

    try:
        result = await dispatch(
            user_input=body.user_input,
            contest_agent=getattr(request.app.state, "contest_agent", None),
            email_agent=getattr(request.app.state, "email_agent", None),
            llm=getattr(request.app.state, "llm", None),
        )
    except Exception as exc:
        log.error("run_agent_error", error=str(exc), user_input=body.user_input[:80])
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    duration_ms = int((time.perf_counter() - t0) * 1000)

    # Write AgentLog (Phase 5) — non-fatal if DB write fails
    try:
        async with get_async_session() as db:
            db.add(AgentLog(
                session_id=session_id,
                user_input=body.user_input[:500],
                agent=result["agent_used"],
                outcome=result["response"][:500],
                duration_ms=duration_ms,
            ))
            await db.commit()
    except Exception as exc:
        log.warning("agent_log_write_failed", error=str(exc))

    return RunResponse(
        response=result["response"],
        agent_used=result["agent_used"],
        pending_action_id=result.get("pending_action_id"),
    )


@app.post(
    "/agent/confirm/{action_id}",
    response_model=ConfirmResponse,
    tags=["HITL"],
    summary="Confirm or cancel a pending HITL action",
)
async def confirm_action(
    action_id: str,
    body: ConfirmRequest,
    request: Request,
) -> ConfirmResponse:
    """
    Resolve a staged pending action.

    - `decision = "confirm"` → executes the action and marks it CONFIRMED.
    - `decision = "undo"`    → marks CANCELLED without executing.

    Error responses:
      404 — action_id not found
      409 — action already resolved (CONFIRMED / CANCELLED)
      410 — action expired (past expires_at)
    """
    hitl: HITLManager = request.app.state.hitl_manager
    try:
        message = await hitl.confirm(action_id, body.decision)
    except ValueError as exc:
        err = str(exc).lower()
        if "not found" in err:
            raise HTTPException(status_code=404, detail=str(exc))
        if "already in state" in err:
            raise HTTPException(status_code=409, detail=str(exc))
        if "expired" in err:
            raise HTTPException(status_code=410, detail=str(exc))
        raise HTTPException(status_code=400, detail=str(exc))

    status = "CONFIRMED" if body.decision == "confirm" else "CANCELLED"
    return ConfirmResponse(action_id=action_id, status=status, message=message)


@app.get(
    "/agent/pending",
    response_model=PendingListResponse,
    tags=["HITL"],
    summary="List all pending HITL actions awaiting user confirmation",
)
async def list_pending(_: Request) -> PendingListResponse:
    """
    Return all PendingAction rows with status = REQUIRES_REVIEW.

    Items are returned in creation order (oldest first).
    """
    async with get_async_session() as session:
        result = await session.execute(
            select(PendingAction)
            .where(PendingAction.status == "REQUIRES_REVIEW")
            .order_by(PendingAction.created_at)
        )
        actions = result.scalars().all()

    def _aware(dt):
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt

    items = [
        PendingActionItem(
            id=a.id,
            action_type=a.action_type,
            description=a.description or "",
            status=a.status,
            expires_at=_aware(a.expires_at),
            created_at=_aware(a.created_at),
        )
        for a in actions
    ]
    return PendingListResponse(pending=items, count=len(items))
