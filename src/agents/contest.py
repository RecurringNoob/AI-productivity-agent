"""
src/agents/contest.py — Contest scheduling agent.

Design Document Reference: Section 8.2 (Contest Agent)

LangGraph: search → select → check_calendar → execute → END

Nodes:
  search:         Fetch upcoming contests via ContestProvider
  select:         LLM picks best contest (ContestSelectionResult)
  check_calendar: Overlap check with travel buffer (read-only, no lock here)
  execute:        Schedule via safe_schedule (acquires lock), or stage HITL if
                  contest duration ≥ HITL_DURATION_HOURS (default 4h)

HITL rule (Design Doc §8.2, Improvement 3):
  Contests ≥ 4 hours are high-stakes → always stage a PendingAction.
  User must confirm via POST /agent/confirm/{id} before event is written.
"""
from __future__ import annotations

import os
from datetime import timedelta
from typing import TypedDict

import structlog
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from src.hitl import HITLManager
from src.llm_client import get_llm
from src.parsers import ConflictInfo, ContestDTO, ContestSelectionResult
from src.tools.calendar import CalendarAdapter
from src.tools.contests import ContestProvider

log = structlog.get_logger(__name__)

HITL_DURATION_HOURS: float = float(os.getenv("HITL_DURATION_HOURS", "4"))


# ==============================================================================
# Graph State
# ==============================================================================


class ContestState(TypedDict):
    user_input:        str
    contests:          list[ContestDTO]
    selected:          ContestSelectionResult | None
    conflict:          ConflictInfo | None
    pending_action_id: str | None
    result:            str


# ==============================================================================
# Agent Class
# ==============================================================================


class ContestAgent:
    """
    Four-node linear LangGraph agent for contest scheduling.

    Adapters and LLM are injected for testability — no singletons used inside nodes.
    """

    def __init__(
        self,
        contest_provider: ContestProvider,
        calendar_adapter: CalendarAdapter,
        hitl_manager: HITLManager,
        llm: BaseChatModel | None = None,
        fetch_limit: int = 5,
    ) -> None:
        self.provider    = contest_provider
        self.calendar    = calendar_adapter
        self.hitl        = hitl_manager
        self.llm         = llm or get_llm()
        self.fetch_limit = fetch_limit
        self._graph      = self._build_graph()

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    async def _search(self, state: ContestState) -> dict:
        """Fetch upcoming contests from all registered providers."""
        contests = await self.provider.fetch(self.fetch_limit)
        log.info("contest_search_complete", count=len(contests))
        return {"contests": contests}

    async def _select(self, state: ContestState) -> dict:
        """Use LLM to pick the most suitable contest (ContestSelectionResult)."""
        contests = state["contests"]
        if not contests:
            return {"selected": None, "result": "No upcoming contests found."}

        contest_list = "\n".join(
            f"- {c.title} ({c.provider})  "
            f"starts {c.start.strftime('%Y-%m-%d %H:%M UTC')}, "
            f"ends {c.end.strftime('%Y-%m-%d %H:%M UTC')}"
            for c in contests
        )
        prompt = (
            f"User request: {state['user_input']}\n\n"
            f"Available upcoming contests:\n{contest_list}\n\n"
            "Pick the most suitable contest. "
            "Return the exact title, start time, duration in hours, and provider."
        )
        try:
            chain = self.llm.with_structured_output(ContestSelectionResult)
            chain = chain.with_retry(stop_after_attempt=2)
            selected: ContestSelectionResult = await chain.ainvoke(prompt)
        except Exception as exc:
            log.error("contest_select_failed", error=str(exc))
            return {"selected": None, "result": f"LLM selection failed: {exc}"}

        log.info("contest_selected", title=selected.title, provider=selected.provider)
        return {"selected": selected}

    async def _check_calendar(self, state: ContestState) -> dict:
        """Read-only overlap check with travel buffer. No event creation here."""
        selected = state.get("selected")
        if selected is None:
            return {"conflict": None}

        end = selected.start + timedelta(hours=selected.duration_hours)
        buffer = int(os.getenv("CALENDAR_TRAVEL_BUFFER_MINUTES", "15"))

        conflict = await self.calendar.check_overlap(selected.start, end, buffer_minutes=buffer)
        return {"conflict": conflict}

    async def _execute(self, state: ContestState) -> dict:
        """
        Atomic schedule via safe_schedule (acquires class-level lock),
        or stage HITL action if contest is high-stakes (≥ HITL_DURATION_HOURS).
        """
        selected = state.get("selected")
        if selected is None:
            return {"result": state.get("result", "No contest selected.")}

        conflict = state.get("conflict")
        if conflict:
            return {
                "result": (
                    f"Cannot schedule '{selected.title}': conflicts with "
                    f"'{conflict.conflicting_title}' "
                    f"({conflict.conflicting_start.strftime('%H:%M')}–"
                    f"{conflict.conflicting_end.strftime('%H:%M')} UTC). "
                    f"Travel buffer applied: {conflict.buffer_applied} min."
                )
            }

        end = selected.start + timedelta(hours=selected.duration_hours)
        external_id = next(
            (c.external_id for c in state["contests"] if c.title == selected.title),
            None,
        )

        # High-stakes: ≥ 4 hours → HITL required (Improvement 3)
        if selected.duration_hours >= HITL_DURATION_HOURS:
            pending_id = await self.hitl.stage(
                action_type="schedule_event",
                payload={
                    "title":       selected.title,
                    "start":       selected.start.isoformat(),
                    "end":         end.isoformat(),
                    "source":      "contest",
                    "priority":    1,
                    "external_id": external_id,
                },
                description=(
                    f"Schedule {selected.duration_hours:.0f}h contest "
                    f"'{selected.title}' on {selected.start.strftime('%Y-%m-%d %H:%M UTC')}"
                ),
            )
            log.info("contest_hitl_staged", title=selected.title, pending_id=pending_id)
            return {
                "pending_action_id": pending_id,
                "result": (
                    f"Found a {selected.duration_hours:.0f}-hour contest "
                    f"'{selected.title}' starting {selected.start.strftime('%Y-%m-%d %H:%M UTC')}. "
                    "Please confirm to add it to your calendar."
                ),
            }

        # Standard (< 4 h): atomic safe_schedule
        result = await self.calendar.safe_schedule(
            title=selected.title,
            start=selected.start,
            end=end,
            source="contest",
            priority=1,
            external_id=external_id,
        )

        if isinstance(result, ConflictInfo):
            return {
                "result": (
                    f"Race condition: '{selected.title}' could not be scheduled — "
                    f"slot taken by '{result.conflicting_title}' during atomic write."
                )
            }

        return {
            "result": (
                f"Scheduled '{result.title}' on "
                f"{result.start_time.strftime('%Y-%m-%d %H:%M')} UTC."
            )
        }

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(ContestState)

        builder.add_node("search",         self._search)
        builder.add_node("select",         self._select)
        builder.add_node("check_calendar", self._check_calendar)
        builder.add_node("execute",        self._execute)

        builder.add_edge(START,            "search")
        builder.add_edge("search",         "select")
        builder.add_edge("select",         "check_calendar")
        builder.add_edge("check_calendar", "execute")
        builder.add_edge("execute",        END)

        return builder.compile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, user_input: str) -> ContestState:
        """Entry point. Returns the final ContestState after all nodes complete."""
        initial: ContestState = {
            "user_input":        user_input,
            "contests":          [],
            "selected":          None,
            "conflict":          None,
            "pending_action_id": None,
            "result":            "",
        }
        return await self._graph.ainvoke(initial)
