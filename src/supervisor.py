"""
src/supervisor.py — Supervisor routing logic.

Design Document Reference: Section 8.1 (Supervisor)

Responsibilities:
  route()    — classify user intent via LLM, apply confidence gate (< 0.6 → clarify)
  dispatch() — route + invoke the appropriate agent, return unified response

Confidence gate (Design Doc §8.1):
  if confidence < 0.6: return clarifying question, do NOT dispatch any agent.
"""
from __future__ import annotations

import structlog
from langchain_core.language_models import BaseChatModel

from src.llm_client import get_llm
from src.parsers import RoutingDecision

log = structlog.get_logger(__name__)

CONFIDENCE_THRESHOLD: float = 0.6

ROUTING_SYSTEM_PROMPT = (
    "You are a routing assistant for an AI productivity agent.\n"
    "Analyze the user's message and determine which agent should handle it.\n\n"
    "Agents:\n"
    "- CONTEST_AGENT: Handles competitive programming contest scheduling "
    "(Codeforces, LeetCode, etc.)\n"
    "- EMAIL_AGENT: Handles email inbox management, triage, and meeting "
    "scheduling from emails\n"
    "- GENERAL: Handles general questions, tasks, and anything else\n\n"
    "Return your classification with a confidence score (0.0 to 1.0)."
)


async def route(
    user_input: str,
    llm: BaseChatModel | None = None,
) -> dict:
    """
    Classify user intent and return a routing decision dict.

    Args:
        user_input: The raw user message.
        llm:        Optional LLM override for testing. Falls back to get_llm().

    Returns:
        {
          "agent":         "CONTEST_AGENT" | "EMAIL_AGENT" | "GENERAL" | "CLARIFY",
          "decision":      RoutingDecision | None,
          "clarification": str | None,       # set only when agent == "CLARIFY"
        }

    Raises:
        RuntimeError if GOOGLE_API_KEY is missing and no llm override is provided.
        Exception propagated from langchain if LLM call fails.
    """
    _llm = llm or get_llm()
    prompt = f"{ROUTING_SYSTEM_PROMPT}\n\nUser message: {user_input}"

    chain = _llm.with_structured_output(RoutingDecision)
    decision: RoutingDecision = await chain.ainvoke(prompt)

    log.info(
        "routing_decision",
        agent=decision.agent,
        confidence=decision.confidence,
        input_preview=user_input[:80],
    )

    if decision.confidence < CONFIDENCE_THRESHOLD:
        return {
            "agent": "CLARIFY",
            "decision": decision,
            "clarification": (
                "Could you clarify your request? I'm not sure whether you need "
                "contest scheduling, email management, or something else."
            ),
        }

    return {"agent": decision.agent, "decision": decision, "clarification": None}


async def dispatch(
    user_input: str,
    contest_agent=None,
    email_agent=None,
    llm: BaseChatModel | None = None,
) -> dict:
    """
    Route the user input and invoke the appropriate agent.

    Args:
        user_input:     Raw user message.
        contest_agent:  ContestAgent instance (injected). None = unavailable.
        email_agent:    EmailAgent instance (injected). None = unavailable.
        llm:            Optional LLM override for testing.

    Returns:
        {
          "response":          str,          — human-readable result
          "agent_used":        str,          — which agent ran
          "pending_action_id": str | None,  — set if HITL was staged
        }
    """
    routing = await route(user_input, llm=llm)
    agent_name = routing["agent"]

    if agent_name == "CLARIFY":
        return {
            "response": routing["clarification"],
            "agent_used": "CLARIFY",
            "pending_action_id": None,
        }

    if agent_name == "CONTEST_AGENT" and contest_agent is not None:
        result = await contest_agent.run(user_input)
        return {
            "response": result.get("result", "Contest agent completed."),
            "agent_used": "CONTEST_AGENT",
            "pending_action_id": result.get("pending_action_id"),
        }

    if agent_name == "EMAIL_AGENT" and email_agent is not None:
        result = await email_agent.run(user_input=user_input)
        report = result.get("action_report", [])
        summary = f"Email triage complete. {len(report)} emails processed." if report else "No emails to process."
        return {
            "response": summary,
            "agent_used": "EMAIL_AGENT",
            "pending_action_id": None,
        }

    if agent_name == "GENERAL":
        _llm = llm or get_llm()
        response = await _llm.ainvoke(user_input)
        return {
            "response": response.content if hasattr(response, "content") else str(response),
            "agent_used": "GENERAL",
            "pending_action_id": None,
        }

    log.warning("dispatch_agent_unavailable", agent=agent_name)
    return {
        "response": f"Agent '{agent_name}' is not currently available.",
        "agent_used": agent_name,
        "pending_action_id": None,
    }
