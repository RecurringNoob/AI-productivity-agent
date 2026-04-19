"""
src/agents/email.py — Email triage agent.

Design Document Reference: Section 8.3 (Email Agent)

LangGraph: fetch → classify_all → reduce → END

Phases:
  fetch:        Retrieve unread emails from EmailAdapter
  classify_all: Classify emails in parallel via asyncio.gather (no DB writes).
                Each email → LLM → EmailClassification → dateparser resolution.
                (Implements Design Doc §8.3 Stage 2 parallelism via asyncio.gather)
  reduce:       Sequential arbitration — processes meeting requests in received_at
                order, acquires calendar lock via safe_schedule, replies or stages HITL.

ARCHITECTURE RULES (Design Doc §5.2):
  1. classify_all nodes NEVER write to DB or calendar.
  2. reduce node is the ONLY writer.
  3. SPAM is NEVER auto-deleted — always staged as HITL (Improvement 3).
  4. Time parsing (Improvement 1): if dateparser returns None → NEEDS_CLARIFICATION.
  5. Travel buffer (Improvement 4): applied in safe_schedule call.
  6. Received-at FIFO ordering for arbitration (Design Doc §8.3).
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone, timedelta
from typing import TypedDict

import structlog
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from src.hitl import HITLManager
from src.llm_client import get_llm
from src.parsers import ConflictInfo, EmailClassification, EmailDTO
from src.tools.calendar import CalendarAdapter
from src.tools.dateparser_util import parse
from src.tools.email import EmailAdapter

log = structlog.get_logger(__name__)

CLASSIFY_PROMPT_TEMPLATE = """Analyze this email and classify its intent.

Email Details:
From: {sender}
Subject: {subject}
Body: {body}

Classification options:
- MEETING_REQUEST: The sender wants to schedule a meeting. Extract meeting_title and meeting_time_raw (exact time phrase from body).
- SPAM: Promotional, unsolicited, or clearly irrelevant email.
- GENERAL: General inquiry or communication (not a meeting request, not spam).
- NEEDS_CLARIFICATION: Meeting request but time is ambiguous or completely unspecified.

If MEETING_REQUEST: extract meeting_title (from subject/body) and meeting_time_raw (raw time string from body).
Return your classification.""".strip()


# ==============================================================================
# Graph State
# ==============================================================================


class EmailState(TypedDict):
    user_input:      str
    user_now:        datetime
    emails:          list[EmailDTO]
    classifications: list[EmailClassification]
    action_report:   list[dict]


# ==============================================================================
# Agent Class
# ==============================================================================


class EmailAgent:
    """
    Three-node LangGraph email triage agent.

    Adapters and LLM are injected for testability.
    """

    def __init__(
        self,
        email_adapter: EmailAdapter,
        calendar_adapter: CalendarAdapter,
        hitl_manager: HITLManager,
        llm: BaseChatModel | None = None,
        user_now: datetime | None = None,
    ) -> None:
        self.email_adapter   = email_adapter
        self.calendar        = calendar_adapter
        self.hitl            = hitl_manager
        self.llm             = llm or get_llm()
        self._user_now       = user_now  # override for testing; defaults to utcnow at runtime
        self._graph          = self._build_graph()

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    async def _fetch(self, state: EmailState) -> dict:
        """Retrieve all unread emails from the EmailAdapter."""
        emails = await self.email_adapter.get_unread()
        log.info("email_fetch_complete", count=len(emails))
        return {"emails": emails}

    async def _classify_all(self, state: EmailState) -> dict:
        """
        Classify all emails in parallel using asyncio.gather (no DB writes).

        For each email:
          1. Call LLM → EmailClassification
          2. Set email_id + received_at from EmailDTO (not from LLM)
          3. Run dateparser on meeting_time_raw (Improvement 1):
             - parseable → set .start, .end
             - None      → change intent to NEEDS_CLARIFICATION

        Failures for individual emails are caught and logged; they are skipped
        rather than aborting the entire batch.
        """
        emails    = state["emails"]
        user_now  = state["user_now"]

        if not emails:
            return {"classifications": []}

        async def classify_one(email: EmailDTO) -> EmailClassification | None:
            prompt = CLASSIFY_PROMPT_TEMPLATE.format(
                sender=email.sender,
                subject=email.subject,
                body=email.body[:1500],  # truncate to avoid token limits
            )
            try:
                chain = self.llm.with_structured_output(EmailClassification)
                classification: EmailClassification = await chain.ainvoke(prompt)
            except Exception as exc:
                log.warning("classify_one_failed", email_id=email.id, error=str(exc))
                return None

            # Attach metadata not produced by LLM
            classification.email_id   = email.id
            classification.received_at = email.received_at

            # Improvement 1: resolve raw time string via dateparser
            if (
                classification.intent == "MEETING_REQUEST"
                and classification.meeting_time_raw
            ):
                meeting_dt = parse(classification.meeting_time_raw, user_now)
                if meeting_dt is None:
                    log.debug(
                        "classify_time_unresolvable",
                        email_id=email.id,
                        raw=classification.meeting_time_raw,
                    )
                    classification.intent = "NEEDS_CLARIFICATION"
                else:
                    classification.start = meeting_dt
                    classification.end   = meeting_dt + timedelta(hours=1)

            return classification

        results = await asyncio.gather(
            *[classify_one(email) for email in emails],
            return_exceptions=True,
        )

        valid = [c for c in results if isinstance(c, EmailClassification)]
        log.info("classify_all_complete", total=len(emails), classified=len(valid))
        return {"classifications": valid}

    async def _reduce(self, state: EmailState) -> dict:
        """
        Sequential arbitration — the ONLY node that writes to DB or sends replies.

        Processing order (Design Doc §8.3):
          1. MEETING_REQUEST sorted by received_at (FIFO arbitration)
             → safe_schedule (acquires lock, applies travel buffer — Improvement 4)
             → if free: reply "Confirmed" + log EmailAction
             → if busy: reply "Busy at that time" + log EmailAction
          2. SPAM → ALWAYS stage HITL (never auto-delete — Improvement 3)
          3. NEEDS_CLARIFICATION → reply asking for clearer time
          4. GENERAL → log only (no automatic reply)
        """
        classifications = state.get("classifications", [])
        if not classifications:
            return {"action_report": []}

        reports: list[dict] = []

        # --- Meeting requests (FIFO by received_at) ---
        meeting_requests = sorted(
            [c for c in classifications if c.intent == "MEETING_REQUEST"],
            key=lambda c: c.received_at or datetime.min.replace(tzinfo=timezone.utc),
        )

        for req in meeting_requests:
            if req.start is None:
                log.warning("reduce_skip_no_start", email_id=req.email_id)
                continue

            end = req.end or (req.start + timedelta(hours=1))

            result = await self.calendar.safe_schedule(
                title=req.meeting_title or f"Meeting (email {req.email_id})",
                start=req.start,
                end=end,
                source="meeting",
                buffer_minutes=req.travel_buffer_minutes,  # Improvement 4
            )

            if isinstance(result, ConflictInfo):
                body = (
                    f"Hi, unfortunately I'm already booked at that time "
                    f"({result.conflicting_start.strftime('%H:%M')}–"
                    f"{result.conflicting_end.strftime('%H:%M')} UTC). "
                    "Could we find another slot?"
                )
                await self.email_adapter.reply(req.email_id, body)
                reports.append({"email_id": req.email_id, "action": "declined"})
                log.info("reduce_meeting_declined", email_id=req.email_id)
            else:
                await self.email_adapter.reply(req.email_id, "Confirmed — added to your calendar.")
                reports.append({"email_id": req.email_id, "action": "scheduled"})
                log.info("reduce_meeting_scheduled", email_id=req.email_id, title=result.title)

        # --- SPAM: always HITL (never auto-delete) ---
        for spam in [c for c in classifications if c.intent == "SPAM"]:
            pending_id = await self.hitl.stage(
                action_type="delete_email",
                payload={"email_id": spam.email_id},
                description=f"Delete spam email (id={spam.email_id})",
            )
            reports.append({
                "email_id":          spam.email_id,
                "action":            "pending_review",
                "pending_action_id": pending_id,
            })
            log.info("reduce_spam_staged", email_id=spam.email_id, pending_id=pending_id)

        # --- NEEDS_CLARIFICATION: ask for clearer time ---
        for clarify in [c for c in classifications if c.intent == "NEEDS_CLARIFICATION"]:
            body = (
                "Hi, I'd love to schedule a meeting, but I wasn't able to parse "
                "the time from your message. Could you specify the date, time, "
                "and timezone? (e.g., '2026-05-15 at 2:00 PM UTC')"
            )
            await self.email_adapter.reply(clarify.email_id, body)
            reports.append({"email_id": clarify.email_id, "action": "clarification_sent"})

        # --- GENERAL: log only ---
        for gen in [c for c in classifications if c.intent == "GENERAL"]:
            reports.append({"email_id": gen.email_id, "action": "logged"})

        return {"action_report": reports}

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(EmailState)

        builder.add_node("fetch",         self._fetch)
        builder.add_node("classify_all",  self._classify_all)
        builder.add_node("reduce",        self._reduce)

        builder.add_edge(START,          "fetch")
        builder.add_edge("fetch",        "classify_all")
        builder.add_edge("classify_all", "reduce")
        builder.add_edge("reduce",       END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, user_input: str = "") -> EmailState:
        """Entry point. Returns the final EmailState after all nodes complete."""
        user_now = self._user_now or datetime.now(timezone.utc)

        initial: EmailState = {
            "user_input":      user_input,
            "user_now":        user_now,
            "emails":          [],
            "classifications": [],
            "action_report":   [],
        }
        return await self._graph.ainvoke(initial)
