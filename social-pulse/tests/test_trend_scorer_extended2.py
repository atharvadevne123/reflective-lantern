"""Extended trend scorer tests with parametrize and boundary coverage."""
from __future__ import annotations

import pytest

from app.trend_scorer import (
    TrendScore,
    avg_score,
    filter_by_tier,
    rank_posts,
    score_post,
    top_n_posts,
)


def _make_score(post_id: str = "P1", trend: float = 0.5, tier: str = "Medium") -> TrendScore:
    return TrendScore(
        post_id=post_id,
        viral_probability=trend,
        engagement_potential=trend,
        trend_score=trend,
        tier=tier,
    )


@pytest.mark.parametrize("viral_prob,followers,hashtags,has_media,mentions", [
    (0.9, 100_000, 3, True, 2),
    (0.1, 100, 0, False, 0),
    (0.5, 10_000, 2, True, 1),
    (0.0, 0, 0, False, 0),
    (1.0, 1_000_000, 5, True, 3),
])
def test_score_post_trend_in_zero_one(
    viral_prob: float, followers: int, hashtags: int, has_media: bool, mentions: int
) -> None:
    result = score_post("test", viral_prob, followers, hashtags, has_media, mentions)
    assert 0.0 <= result.trend_score <= 1.0


@pytest.mark.parametrize("score,expected_tier", [
    (0.0, "Low"),
    (0.24, "Low"),
    (0.25, "Medium"),
    (0.49, "Medium"),
    (0.50, "High"),
    (0.74, "High"),
    (0.75, "Viral"),
    (1.0, "Viral"),
])
def test_score_post_tier_thresholds(score: float, expected_tier: str) -> None:
    result = score_post("x", score, 0, 0, False, 0, probability_weight=1.0)
    assert result.tier == expected_tier


def test_score_post_returns_trend_score() -> None:
    result = score_post("p1", 0.8, 50_000, 3, True, 1)
    assert isinstance(result, TrendScore)
    assert result.post_id == "p1"


def test_score_post_zero_followers() -> None:
    result = score_post("p0", 0.5, 0, 0, False, 0)
    assert isinstance(result.trend_score, float)
    assert 0.0 <= result.trend_score <= 1.0


def test_score_post_viral_prob_one_is_viral() -> None:
    result = score_post("pmax", 1.0, 1_000_000, 5, True, 10, probability_weight=0.9)
    assert result.tier == "Viral"


def test_rank_posts_descending_order() -> None:
    scores = [
        _make_score("A", 0.3),
        _make_score("B", 0.9),
        _make_score("C", 0.1),
        _make_score("D", 0.7),
    ]
    ranked = rank_posts(scores)
    for i in range(len(ranked) - 1):
        assert ranked[i].trend_score >= ranked[i + 1].trend_score


def test_rank_posts_empty_list() -> None:
    assert rank_posts([]) == []


def test_rank_posts_single_item() -> None:
    scores = [_make_score("only", 0.5)]
    ranked = rank_posts(scores)
    assert len(ranked) == 1


@pytest.mark.parametrize("n", [1, 3, 5, 10])
def test_top_n_posts_at_most_n(n: int) -> None:
    scores = [_make_score(str(i), float(i) / 20.0) for i in range(15)]
    result = top_n_posts(scores, n=n)
    assert len(result) <= n


def test_top_n_posts_empty_input() -> None:
    assert top_n_posts([], n=5) == []


@pytest.mark.parametrize("tier", ["Viral", "High", "Medium", "Low"])
def test_filter_by_tier_returns_only_matching(tier: str) -> None:
    scores = [
        _make_score("v1", 0.9, "Viral"),
        _make_score("h1", 0.6, "High"),
        _make_score("m1", 0.4, "Medium"),
        _make_score("l1", 0.1, "Low"),
    ]
    result = filter_by_tier(scores, tier)
    assert all(s.tier == tier for s in result)


def test_filter_by_tier_case_insensitive() -> None:
    scores = [_make_score("v1", 0.9, "Viral")]
    assert len(filter_by_tier(scores, "viral")) == 1
    assert len(filter_by_tier(scores, "VIRAL")) == 1


def test_avg_score_empty_returns_zero() -> None:
    assert avg_score([]) == 0.0


def test_avg_score_single_item() -> None:
    scores = [_make_score("x", 0.6)]
    assert avg_score(scores) == pytest.approx(0.6, rel=1e-6)


def test_avg_score_multiple_items() -> None:
    scores = [_make_score("a", 0.2), _make_score("b", 0.8)]
    assert avg_score(scores) == pytest.approx(0.5, rel=1e-6)


def test_trend_score_to_dict_has_required_keys() -> None:
    result = score_post("p1", 0.7, 10_000, 2, True, 1)
    d = result.to_dict()
    for key in ("post_id", "viral_probability", "engagement_potential", "trend_score", "tier"):
        assert key in d
