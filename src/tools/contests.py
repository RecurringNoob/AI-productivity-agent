"""
src/tools/contests.py — ContestProvider ABC and provider implementations.

Design Document Reference:
  Section 9.4 (Contest Provider), Improvement 5 (Levenshtein deduplication)

Providers:
  CodeforcesProvider        — Fetches upcoming contests from Codeforces public API.
  LeetCodeProvider          — Fetches upcoming contests from LeetCode GraphQL API.
  AggregatingContestProvider — Fans out to all providers concurrently, deduplicates
                               near-identical titles using Levenshtein distance.

Deduplication (Improvement 5):
  After merging all provider results, _similar() finds titles within edit-distance 5
  that also start within 1 hour of each other and keeps only the first occurrence
  (sorted by start time). The UNIQUE(external_id) constraint in the DB is the
  second line of defense in case the semantic dedup misses anything.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import List

import httpx
import structlog
from Levenshtein import distance as levenshtein_distance

from src.parsers import ContestDTO

log = structlog.get_logger(__name__)


# ==============================================================================
# Abstract Base Class
# ==============================================================================


class ContestProvider(ABC):
    """Abstract interface for all contest data sources."""

    @abstractmethod
    async def fetch(self, limit: int) -> List[ContestDTO]:
        """Return up to `limit` upcoming contests, sorted by start time."""
        ...


# ==============================================================================
# Levenshtein helper (Improvement 5)
# ==============================================================================


def _similar(a: str, b: str, threshold: int = 5) -> bool:
    """
    Return True if the two lowercased title strings are within `threshold`
    edit operations of each other.

    Used by AggregatingContestProvider to identify near-duplicate contest titles
    that appear in multiple providers (e.g., the same contest listed under
    slightly different names on Codeforces and LeetCode mirrors).

    Design Doc §9.4: threshold=5 is the default; tests must not depend on
    the internal value.
    """
    return levenshtein_distance(a.lower(), b.lower()) <= threshold


# ==============================================================================
# Codeforces Provider
# ==============================================================================


class CodeforcesProvider(ContestProvider):
    """
    Fetches upcoming contests from the public Codeforces REST API.
    Endpoint: GET https://codeforces.com/api/contest.list
    Filters:  phase == "BEFORE" (upcoming only)
    """

    API_URL = "https://codeforces.com/api/contest.list"

    async def fetch(self, limit: int) -> List[ContestDTO]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(self.API_URL)
                resp.raise_for_status()
                data = resp.json()

            if data.get("status") != "OK":
                log.warning("codeforces_api_bad_status", status=data.get("status"))
                return []

            contests: List[ContestDTO] = []
            for c in data.get("result", []):
                if c.get("phase") != "BEFORE":
                    continue  # Skip ongoing and finished contests

                start_ts = c.get("startTimeSeconds")
                if start_ts is None:
                    continue

                start = datetime.fromtimestamp(start_ts, tz=timezone.utc)
                end   = start + timedelta(seconds=c.get("durationSeconds", 7200))

                contests.append(
                    ContestDTO(
                        title=c["name"],
                        start=start,
                        end=end,
                        provider="codeforces",
                        external_id=f"codeforces-{c['id']}",
                    )
                )

            return sorted(contests, key=lambda x: x.start)[:limit]

        except Exception as exc:
            log.warning("codeforces_provider_failed", error=str(exc))
            return []


# ==============================================================================
# LeetCode Provider
# ==============================================================================


class LeetCodeProvider(ContestProvider):
    """
    Fetches upcoming contests from the LeetCode GraphQL API.
    Endpoint: POST https://leetcode.com/graphql/
    Query: allContests { title, titleSlug, startTime, duration }
    Filters: startTime > now (upcoming only)
    """

    GRAPHQL_URL = "https://leetcode.com/graphql/"
    QUERY = """
    query {
        allContests {
            title
            titleSlug
            startTime
            duration
        }
    }
    """

    async def fetch(self, limit: int) -> List[ContestDTO]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    self.GRAPHQL_URL,
                    json={"query": self.QUERY},
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": (
                            "Mozilla/5.0 (compatible; AI-Productivity-Agent/3.0)"
                        ),
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            raw_contests = (data.get("data") or {}).get("allContests") or []
            now_ts = datetime.now(timezone.utc).timestamp()

            contests: List[ContestDTO] = []
            for c in raw_contests:
                start_ts = c.get("startTime", 0)
                if start_ts <= now_ts:
                    continue  # skip past contests

                start = datetime.fromtimestamp(start_ts, tz=timezone.utc)
                end   = start + timedelta(seconds=c.get("duration", 5400))

                slug = c.get("titleSlug") or c.get("title", "unknown").lower().replace(" ", "-")
                contests.append(
                    ContestDTO(
                        title=c["title"],
                        start=start,
                        end=end,
                        provider="leetcode",
                        external_id=f"leetcode-{slug}",
                    )
                )

            return sorted(contests, key=lambda x: x.start)[:limit]

        except Exception as exc:
            log.warning("leetcode_provider_failed", error=str(exc))
            return []


# ==============================================================================
# Aggregating Provider (Improvement 5)
# ==============================================================================


class AggregatingContestProvider(ContestProvider):
    """
    Aggregates results from all registered ContestProviders concurrently using
    asyncio.gather, then deduplicates near-identical titles.

    Deduplication algorithm (Improvement 5):
    1. Sort all contests by start time.
    2. For each candidate, check if any already-accepted contest has:
       - A title within Levenshtein distance threshold (default 5), AND
       - A start time within 1 hour.
    3. If both conditions match, treat as a duplicate and skip.
    4. The UNIQUE(external_id) DB constraint acts as a hard fallback.

    Provider failures are isolated: return_exceptions=True ensures one provider
    being unreachable does not abort the others. (Design Doc §10)
    """

    def __init__(self, providers: List[ContestProvider]) -> None:
        self.providers = providers

    async def fetch(self, limit: int) -> List[ContestDTO]:
        results = await asyncio.gather(
            *[p.fetch(limit) for p in self.providers],
            return_exceptions=True,
        )

        all_contests: List[ContestDTO] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.warning(
                    "aggregating_provider_partial_failure",
                    provider_index=i,
                    error=str(result),
                )
            else:
                all_contests.extend(result)

        # Levenshtein deduplication (Improvement 5)
        deduped: List[ContestDTO] = []
        for candidate in sorted(all_contests, key=lambda c: c.start):
            is_duplicate = any(
                _similar(candidate.title, seen.title)
                and abs((candidate.start - seen.start).total_seconds()) < 3600
                for seen in deduped
            )
            if not is_duplicate:
                deduped.append(candidate)

        log.info(
            "aggregating_fetch_complete",
            total_before_dedup=len(all_contests),
            total_after_dedup=len(deduped),
        )

        return deduped[:limit]
