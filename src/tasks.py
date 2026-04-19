"""
src/tasks.py — Background asyncio tasks.

Design Document Reference: §9.5 (HITL Manager — periodic expiry, Improvement 3)

Contains:
  expire_stale_loop — calls HITLManager.expire_stale() at a configurable interval.
    Started as an asyncio.Task in the FastAPI lifespan (Phase 5).
    Cancelled gracefully on server shutdown via task.cancel() + await.

No extra scheduler library needed — a bare asyncio.Task is sufficient and
keeps the dependency footprint minimal.
"""
from __future__ import annotations

import asyncio

import structlog

log = structlog.get_logger(__name__)


async def expire_stale_loop(hitl_manager, interval_seconds: int = 300) -> None:
    """
    Infinite loop that calls hitl_manager.expire_stale() every interval_seconds.

    Runs until cancelled. On asyncio.CancelledError: logs that the loop stopped,
    then re-raises so the Task terminates cleanly without swallowing the signal.

    Args:
        hitl_manager:      HITLManager instance injected from app.state.
        interval_seconds:  Poll interval in seconds (default 300 = 5 min).
                           Configurable via HITL_EXPIRY_CHECK_SECONDS env var.
    """
    log.info("expire_stale_loop_started", interval_seconds=interval_seconds)
    try:
        while True:
            await asyncio.sleep(interval_seconds)
            count = await hitl_manager.expire_stale()
            if count:
                log.info("expire_stale_loop_tick", expired_count=count)
    except asyncio.CancelledError:
        log.info("expire_stale_loop_stopped")
        raise
