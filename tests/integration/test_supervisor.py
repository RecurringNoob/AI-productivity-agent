"""
tests/integration/test_supervisor.py — Integration tests for src/supervisor.py

Uses mock LLM — no real Gemini API calls.

Exit criteria (Phase 2, Design Doc §8.1):
  ✓ CONTEST_AGENT routing with high confidence
  ✓ EMAIL_AGENT routing with high confidence
  ✓ GENERAL routing
  ✓ Confidence < 0.6 → "CLARIFY" with a clarification message
  ✓ Clarification message is non-empty string
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.parsers import RoutingDecision
from src.supervisor import CONFIDENCE_THRESHOLD, route


def make_mock_llm(agent: str, confidence: float) -> MagicMock:
    """Build a mock LLM that returns the given RoutingDecision."""
    mock_llm = MagicMock()
    mock_chain = AsyncMock()
    mock_chain.ainvoke = AsyncMock(
        return_value=RoutingDecision(agent=agent, confidence=confidence)
    )
    mock_llm.with_structured_output.return_value = mock_chain
    return mock_llm


class TestRoute:

    async def test_routes_to_contest_agent(self):
        llm = make_mock_llm("CONTEST_AGENT", 0.95)
        result = await route("find me a codeforces contest", llm=llm)
        assert result["agent"] == "CONTEST_AGENT"
        assert result["clarification"] is None
        assert isinstance(result["decision"], RoutingDecision)

    async def test_routes_to_email_agent(self):
        llm = make_mock_llm("EMAIL_AGENT", 0.88)
        result = await route("check my emails", llm=llm)
        assert result["agent"] == "EMAIL_AGENT"
        assert result["clarification"] is None

    async def test_routes_to_general(self):
        llm = make_mock_llm("GENERAL", 0.75)
        result = await route("what is the capital of France?", llm=llm)
        assert result["agent"] == "GENERAL"
        assert result["clarification"] is None

    async def test_low_confidence_returns_clarify(self):
        """Confidence below threshold → CLARIFY, never dispatch."""
        llm = make_mock_llm("CONTEST_AGENT", CONFIDENCE_THRESHOLD - 0.01)
        result = await route("something unclear maybe", llm=llm)
        assert result["agent"] == "CLARIFY"
        assert result["clarification"] is not None
        assert len(result["clarification"]) > 0

    async def test_exactly_at_threshold_is_not_clarify(self):
        """Confidence exactly equal to threshold should dispatch (not clarify)."""
        llm = make_mock_llm("GENERAL", CONFIDENCE_THRESHOLD)
        result = await route("something", llm=llm)
        assert result["agent"] == "GENERAL"

    async def test_zero_confidence_returns_clarify(self):
        llm = make_mock_llm("GENERAL", 0.0)
        result = await route("???", llm=llm)
        assert result["agent"] == "CLARIFY"

    async def test_max_confidence_routes_normally(self):
        llm = make_mock_llm("EMAIL_AGENT", 1.0)
        result = await route("process my inbox", llm=llm)
        assert result["agent"] == "EMAIL_AGENT"

    async def test_with_structured_output_called_with_routing_decision(self):
        """Ensure the supervisor calls with_structured_output(RoutingDecision)."""
        llm = make_mock_llm("GENERAL", 0.8)
        await route("test input", llm=llm)
        llm.with_structured_output.assert_called_once_with(RoutingDecision)
