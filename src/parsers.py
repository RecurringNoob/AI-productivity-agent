"""
src/parsers.py — All Pydantic schemas for LLM outputs and inter-layer DTOs.

Design Document Reference:
  §6.2 — Pydantic LLM Output Schemas (RoutingDecision, ContestSelectionResult, EmailClassification)
  §6.3 — DTOs (ContestDTO, EmailDTO, ConflictInfo)

ARCHITECTURE RULE: ALL LLM structured output schemas are defined here.
No schema definitions are allowed in agent or tool files.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ==============================================================================
# LLM Output Schemas — used with llm.with_structured_output(SchemaClass)
# Design Document §6.2
# ==============================================================================


class RoutingDecision(BaseModel):
    """
    Supervisor output — determines which sub-agent to dispatch to.

    If confidence < 0.6, the supervisor returns a clarifying question
    rather than routing. (Design Doc §8.1)
    """
    agent: Literal["CONTEST_AGENT", "EMAIL_AGENT", "GENERAL"]
    confidence: float = Field(ge=0.0, le=1.0)


class ContestSelectionResult(BaseModel):
    """
    Contest agent 'select' node output.
    Identifies which contest from the fetched list should be scheduled.
    """
    title: str
    start: datetime
    duration_hours: float
    provider: str                    # "codeforces" | "leetcode"


class EmailClassification(BaseModel):
    """
    Email agent 'classify_only' node output — one per email, produced in parallel.

    Fields:
    - email_id:              Linked back to EmailDTO.id (set by agent, not LLM)
    - intent:                LLM-classified intent
    - meeting_title:         If MEETING_REQUEST — extracted meeting title
    - meeting_time_raw:      If MEETING_REQUEST — raw time string from body
    - travel_buffer_minutes: Soft-conflict buffer (Improvement 4, default 15)
    - received_at:           Carried from EmailDTO.received_at for reduce node ordering
    - start / end:           Resolved datetime objects (set after dateparser, not by LLM)
    """
    email_id: str | None = None
    intent: Literal["MEETING_REQUEST", "SPAM", "GENERAL", "NEEDS_CLARIFICATION"]
    meeting_title: str | None = None
    meeting_time_raw: str | None = None     # Raw string from email body (Improvement 1)
    travel_buffer_minutes: int = Field(default=15)  # Improvement 4 — configurable buffer
    received_at: datetime | None = None     # Carried from EmailDTO for arbitration ordering
    # Resolved after dateparser.parse — NOT set by LLM
    start: datetime | None = None
    end: datetime | None = None


# ==============================================================================
# DTOs — Data Transfer Objects between tool and agent layers
# Design Document §6.3
# ==============================================================================


class ContestDTO(BaseModel):
    """
    Contest data returned by any ContestProvider implementation.
    Used by the contest agent 'search' node and passed to 'select'.
    """
    title: str
    start: datetime
    end: datetime
    provider: str           # "codeforces" | "leetcode"
    external_id: str        # Enforced by UNIQUE constraint on events.external_id


class EmailDTO(BaseModel):
    """
    Email data returned by any EmailAdapter implementation.
    received_at is used for FIFO arbitration in the email reduce node.
    (Design Doc §5.2 — Phase 2)
    """
    id: str
    sender: str
    subject: str
    body: str
    received_at: datetime   # Used for arbitration ordering in Phase 2 reduce node


class ConflictInfo(BaseModel):
    """
    Returned by CalendarAdapter.check_overlap when a scheduling conflict exists.
    buffer_applied is included for transparency in the API response.
    (Design Doc §6.3, §9.1)
    """
    conflicting_title: str
    conflicting_start: datetime
    conflicting_end: datetime
    buffer_applied: int     # Minutes — shows the user which buffer triggered the conflict
