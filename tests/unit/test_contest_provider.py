"""
tests/unit/test_contest_provider.py — Unit tests for ContestProvider implementations.

Uses AsyncMock to stub individual providers so no real HTTP calls are made.

Exit criteria (Phase 1, Design Doc):
  ✓ AggregatingContestProvider de-duplicates a pair of near-identical contest titles
  ✓ One provider failing does not abort the others
  ✓ Results are sorted by start time
  ✓ limit is respected
  ✓ _similar() is case-insensitive and threshold-aware
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock

import pytest

from src.parsers import ContestDTO
from src.tools.contests import AggregatingContestProvider, _similar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_dto(
    title: str,
    start: datetime,
    provider: str = "codeforces",
    offset_id: int = 0,
) -> ContestDTO:
    return ContestDTO(
        title=title,
        start=start,
        end=start + timedelta(hours=2),
        provider=provider,
        external_id=f"{provider}-{offset_id}",
    )


T0 = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# _similar() helper tests (Improvement 5)
# ---------------------------------------------------------------------------


class TestSimilarHelper:

    def test_identical_strings_are_similar(self):
        assert _similar("Codeforces Round 987", "Codeforces Round 987")

    def test_completely_different_strings_are_not_similar(self):
        assert not _similar("Codeforces Round 987", "LeetCode Weekly 350")

    def test_case_insensitive(self):
        assert _similar("CODEFORCES ROUND 987", "codeforces round 987")

    def test_near_identical_within_threshold(self):
        # One extra space — distance of 1 → similar
        assert _similar("Codeforces Round 987 (Div. 2)", "Codeforces Round 987 (Div. 2 )")

    def test_strings_beyond_threshold_not_similar(self):
        # Very different titles should have high distance
        assert not _similar("CF 987", "LeetCode Biweekly Contest 150")

    def test_empty_strings_are_similar(self):
        assert _similar("", "")


# ---------------------------------------------------------------------------
# AggregatingContestProvider tests
# ---------------------------------------------------------------------------


class TestAggregatingContestProvider:

    async def test_merges_results_from_two_providers(self):
        mock_cf = AsyncMock()
        mock_cf.fetch.return_value = [make_dto("CF Round 1", T0, "codeforces", 1)]

        mock_lc = AsyncMock()
        mock_lc.fetch.return_value = [
            make_dto("LC Weekly 1", T0 + timedelta(hours=3), "leetcode", 1)
        ]

        provider = AggregatingContestProvider([mock_cf, mock_lc])
        result = await provider.fetch(10)

        assert len(result) == 2
        assert {c.provider for c in result} == {"codeforces", "leetcode"}

    async def test_deduplicates_near_identical_titles_same_time(self):
        """
        Improvement 5: near-identical title + same start time → 1 result.
        Both providers report essentially the same contest.
        """
        mock_cf = AsyncMock()
        mock_cf.fetch.return_value = [
            make_dto("Codeforces Round 987 (Div. 2)", T0, "codeforces", 1)
        ]
        mock_lc = AsyncMock()
        mock_lc.fetch.return_value = [
            make_dto("Codeforces Round 987 (Div. 2 )", T0, "leetcode", 2)  # 1 char diff
        ]

        provider = AggregatingContestProvider([mock_cf, mock_lc])
        result = await provider.fetch(10)

        assert len(result) == 1

    async def test_does_not_deduplicate_similar_titles_far_apart_in_time(self):
        """
        Same title but start times > 1 hour apart → NOT duplicates.
        These are legitimately different contests with similar names.
        """
        mock_cf = AsyncMock()
        mock_cf.fetch.return_value = [
            make_dto("Weekly Contest 400", T0, "codeforces", 1)
        ]
        mock_lc = AsyncMock()
        mock_lc.fetch.return_value = [
            make_dto("Weekly Contest 400", T0 + timedelta(hours=2), "leetcode", 2)
        ]

        provider = AggregatingContestProvider([mock_cf, mock_lc])
        result = await provider.fetch(10)

        assert len(result) == 2

    async def test_provider_failure_does_not_abort_others(self):
        """
        Design Doc §10: Codeforces API unreachable → warning logged,
        other providers still run. (return_exceptions=True)
        """
        mock_cf = AsyncMock()
        mock_cf.fetch.side_effect = Exception("Connection refused")

        mock_lc = AsyncMock()
        mock_lc.fetch.return_value = [make_dto("LC Weekly 1", T0, "leetcode", 1)]

        provider = AggregatingContestProvider([mock_cf, mock_lc])
        result = await provider.fetch(10)

        assert len(result) == 1
        assert result[0].provider == "leetcode"

    async def test_both_providers_fail_returns_empty(self):
        mock_cf = AsyncMock()
        mock_cf.fetch.side_effect = Exception("CF down")
        mock_lc = AsyncMock()
        mock_lc.fetch.side_effect = Exception("LC down")

        provider = AggregatingContestProvider([mock_cf, mock_lc])
        result = await provider.fetch(10)

        assert result == []

    async def test_respects_limit(self):
        mock_cf = AsyncMock()
        mock_cf.fetch.return_value = [
            make_dto(f"CF Round {i}", T0 + timedelta(days=i), "codeforces", i)
            for i in range(10)
        ]
        mock_lc = AsyncMock()
        mock_lc.fetch.return_value = []

        provider = AggregatingContestProvider([mock_cf, mock_lc])
        result = await provider.fetch(3)

        assert len(result) == 3

    async def test_results_sorted_by_start_time(self):
        """Result must always be ordered ascending by start time."""
        mock_cf = AsyncMock()
        mock_cf.fetch.return_value = [
            make_dto("CF Round 2", T0 + timedelta(days=2), "codeforces", 2),
            make_dto("CF Round 1", T0 + timedelta(days=1), "codeforces", 1),
        ]
        mock_lc = AsyncMock()
        mock_lc.fetch.return_value = []

        provider = AggregatingContestProvider([mock_cf, mock_lc])
        result = await provider.fetch(10)

        assert result[0].title == "CF Round 1"
        assert result[1].title == "CF Round 2"

    async def test_empty_providers_list_returns_empty(self):
        provider = AggregatingContestProvider([])
        result = await provider.fetch(10)
        assert result == []
