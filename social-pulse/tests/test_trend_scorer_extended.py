"""Extended tests for social-pulse app/trend_scorer.py."""

from __future__ import annotations

import pytest
from app.trend_scorer import TrendScore, rank_posts, score_post


class TestTrendScore:
    def test_to_dict_has_required_keys(self) -> None:
        ts = TrendScore(
            post_id="p1",
            viral_probability=0.8,
            engagement_potential=0.6,
            trend_score=0.72,
            tier="High",
        )
        d = ts.to_dict()
        assert "post_id" in d
        assert "viral_probability" in d
        assert "engagement_potential" in d
        assert "trend_score" in d
        assert "tier" in d

    def test_to_dict_rounds_floats(self) -> None:
        ts = TrendScore(
            post_id="p1",
            viral_probability=0.123456789,
            engagement_potential=0.5,
            trend_score=0.6,
            tier="High",
        )
        d = ts.to_dict()
        assert len(str(d["viral_probability"]).replace("0.", "")) <= 6


class TestScorePost:
    def test_returns_trend_score(self) -> None:
        result = score_post("p1", 0.8, 10000, 3, True, 2)
        assert isinstance(result, TrendScore)

    def test_trend_score_in_range(self) -> None:
        result = score_post("p1", 0.5, 5000, 2, False, 1)
        assert 0.0 <= result.trend_score <= 1.0

    def test_viral_probability_zero_gives_low_score(self) -> None:
        result = score_post("p1", 0.0, 0, 0, False, 0)
        assert result.trend_score == pytest.approx(0.0)

    def test_high_probability_high_score(self) -> None:
        result = score_post("p1", 1.0, 1000000, 10, True, 5)
        assert result.trend_score >= 0.7

    def test_tier_viral_threshold(self) -> None:
        result = score_post("p1", 0.99, 1000000, 10, True, 5)
        assert result.tier == "Viral"

    def test_tier_low_threshold(self) -> None:
        result = score_post("p1", 0.0, 0, 0, False, 0)
        assert result.tier == "Low"

    def test_post_id_preserved(self) -> None:
        result = score_post("my-unique-post-123", 0.5, 1000, 2, True, 1)
        assert result.post_id == "my-unique-post-123"

    def test_probability_weight_affects_score(self) -> None:
        r1 = score_post("p1", 0.9, 0, 0, False, 0, probability_weight=0.9)
        r2 = score_post("p1", 0.9, 0, 0, False, 0, probability_weight=0.1)
        assert r1.trend_score != r2.trend_score


class TestRankPosts:
    def test_empty_list_returns_empty(self) -> None:
        assert rank_posts([]) == []

    def test_sorted_descending(self) -> None:
        scores = [
            TrendScore("p1", 0.3, 0.3, 0.3, "Low"),
            TrendScore("p2", 0.8, 0.8, 0.8, "Viral"),
            TrendScore("p3", 0.5, 0.5, 0.5, "Medium"),
        ]
        ranked = rank_posts(scores)
        values = [s.trend_score for s in ranked]
        assert values == sorted(values, reverse=True)

    def test_single_item_returned(self) -> None:
        ts = TrendScore("p1", 0.5, 0.5, 0.5, "Medium")
        result = rank_posts([ts])
        assert len(result) == 1
        assert result[0].post_id == "p1"
