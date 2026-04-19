"""
src/api/schemas.py — Pydantic request / response models for the REST API.

Design Document Reference: Section 10 (API Contract)

All HTTP-level validation is handled here. Domain schemas (RoutingDecision etc.)
live in src/parsers.py and are NEVER exposed directly as API response types.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ==============================================================================
# POST /agent/run
# ==============================================================================


class RunRequest(BaseModel):
    """Body for POST /agent/run."""

    user_input: str = Field(
        min_length=1,
        max_length=2000,
        description="The user's natural-language request.",
    )
    user_timezone: str = Field(
        default="UTC",
        description="IANA timezone string, e.g. 'America/New_York'. Used for dateparser.",
    )

    model_config = {"json_schema_extra": {"example": {"user_input": "Find me a Codeforces contest this weekend"}}}


class RunResponse(BaseModel):
    """Response for POST /agent/run."""

    response:          str
    agent_used:        str
    pending_action_id: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "response": "Scheduled 'CF Round 987' on 2026-05-10 14:00 UTC.",
                "agent_used": "CONTEST_AGENT",
                "pending_action_id": None,
            }
        }
    }


# ==============================================================================
# POST /agent/confirm/{action_id}
# ==============================================================================


class ConfirmRequest(BaseModel):
    """Body for POST /agent/confirm/{action_id}."""

    decision: Literal["confirm", "undo"] = Field(
        description="'confirm' executes the staged action; 'undo' cancels it."
    )


class ConfirmResponse(BaseModel):
    """Response for POST /agent/confirm/{action_id}."""

    action_id: str
    status:    str   # CONFIRMED | CANCELLED | EXPIRED
    message:   str


# ==============================================================================
# GET /agent/pending
# ==============================================================================


class PendingActionItem(BaseModel):
    """Single pending action entry in the list response."""

    id:          str
    action_type: str
    description: str
    status:      str
    expires_at:  datetime
    created_at:  datetime


class PendingListResponse(BaseModel):
    """Response for GET /agent/pending."""

    pending: list[PendingActionItem]
    count:   int


# ==============================================================================
# Error response (used by HTTPException handlers, informational only)
# ==============================================================================


class ErrorResponse(BaseModel):
    detail: str


# ==============================================================================
# GET /metrics
# ==============================================================================


class MetricsResponse(BaseModel):
    """Response for GET /metrics — lightweight operational snapshot."""

    pending_requires_review: int = Field(description="Actions awaiting user confirmation.")
    pending_expired:         int = Field(description="Actions that expired without confirmation.")
    events_total:            int = Field(description="Total calendar events in the DB.")

