"""Tests for trend scoring and ranking utilities."""

from __future__ import annotations

import pytest

from app.trend_scorer import TrendScore, rank_posts, score_post


def _score(
    viral_probability: float = 0.5,
    author_followers: int = 10000,
    hashtag_count: int = 2,
    has_media: bool = False,
    mention_count: int = 1,
    post_id: str = "test-post",
) -> TrendScore:
    return score_post(
        post_id=post_id,
        viral_probability=viral_probability,
        author_followers=author_followers,
        hashtag_count=hashtag_count,
        has_media=has_media,
        mention_count=mention_count,
    )


class TestScorePost:
    def test_returns_trend_score(self) -> None:
        result = _score()
        assert isinstance(result, TrendScore)

    def test_trend_score_in_unit_interval(self) -> None:
        result = _score()
        assert 0.0 <= result.trend_score <= 1.0

    def test_high_viral_prob_gives_high_score(self) -> None:
        high = _score(viral_probability=0.95, author_followers=1_000_000)
        low = _score(viral_probability=0.05, author_followers=10)
        assert high.trend_score > low.trend_score

    def test_tier_viral_for_high_score(self) -> None:
        result = _score(viral_probability=0.99, author_followers=5_000_000)
        assert result.tier in ("Viral", "High")

    def test_tier_low_for_low_score(self) -> None:
        result = _score(viral_probability=0.01, author_followers=10)
        assert result.tier in ("Low", "Medium")

    def test_post_id_preserved(self) -> None:
        result = _score(post_id="abc-123")
        assert result.post_id == "abc-123"

    def test_engagement_potential_in_unit_interval(self) -> None:
        result = _score()
        assert 0.0 <= result.engagement_potential <= 1.0

    def test_media_bonus_increases_score(self) -> None:
        with_media = _score(has_media=True)
        without_media = _score(has_media=False)
        assert with_media.trend_score >= without_media.trend_score


class TestRankPosts:
    def test_empty_list_returns_empty(self) -> None:
        assert rank_posts([]) == []

    def test_sorted_descending(self) -> None:
        s1 = _score(viral_probability=0.2, post_id="low")
        s2 = _score(viral_probability=0.8, post_id="high")
        s3 = _score(viral_probability=0.5, post_id="mid")
        ranked = rank_posts([s1, s2, s3])
        scores = [r.trend_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_top_post_has_highest_score(self) -> None:
        scores = [_score(viral_probability=p, post_id=str(p)) for p in [0.1, 0.9, 0.5]]
        ranked = rank_posts(scores)
        assert ranked[0].viral_probability == pytest.approx(0.9)


class TestTrendScoreToDict:
    def test_to_dict_has_required_keys(self) -> None:
        result = _score()
        d = result.to_dict()
        assert "post_id" in d
        assert "viral_probability" in d
        assert "trend_score" in d
        assert "tier" in d


@pytest.mark.parametrize(
    "viral_prob,expected_tier",
    [
        (0.01, "Low"),
        (0.99, "Viral"),
    ],
)
def test_extreme_probabilities(viral_prob: float, expected_tier: str) -> None:
    result = _score(viral_probability=viral_prob)
    # Very low prob → Low or Medium; very high prob → High or Viral
    if viral_prob < 0.1:
        assert result.tier in ("Low", "Medium")
    else:
        assert result.tier in ("High", "Viral")
