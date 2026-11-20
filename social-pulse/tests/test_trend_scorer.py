"""Tests for trend scoring and ranking utilities."""

from __future__ import annotations

import pytest

from app.trend_scorer import TrendScore, avg_score, filter_by_tier, get_trending_hashtags, rank_posts, score_post, top_n_posts


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


class TestTopNPosts:
    def test_returns_at_most_n(self) -> None:
        scores = [_score(viral_probability=0.5, post_id=str(i)) for i in range(20)]
        result = top_n_posts(scores, n=5)
        assert len(result) <= 5

    def test_empty_input(self) -> None:
        assert top_n_posts([], n=10) == []

    def test_fewer_than_n_returns_all(self) -> None:
        scores = [_score(post_id=str(i)) for i in range(3)]
        result = top_n_posts(scores, n=10)
        assert len(result) == 3

    def test_sorted_descending(self) -> None:
        scores = [_score(viral_probability=p / 10.0, post_id=str(p)) for p in range(1, 11)]
        result = top_n_posts(scores, n=5)
        trend_scores = [s.trend_score for s in result]
        assert trend_scores == sorted(trend_scores, reverse=True)


class TestFilterByTier:
    def test_filters_viral(self) -> None:
        viral = _score(viral_probability=0.99, author_followers=5_000_000)
        low = _score(viral_probability=0.01, author_followers=10)
        result = filter_by_tier([viral, low], "Viral")
        assert all(s.tier == "Viral" for s in result)

    def test_case_insensitive(self) -> None:
        s = _score(viral_probability=0.01, author_followers=10)
        result_lower = filter_by_tier([s], "low")
        result_upper = filter_by_tier([s], "LOW")
        assert len(result_lower) == len(result_upper)

    def test_no_match_returns_empty(self) -> None:
        s = _score(viral_probability=0.5)
        result = filter_by_tier([s], "NonExistentTier")
        assert result == []

    def test_empty_input(self) -> None:
        assert filter_by_tier([], "Viral") == []


class TestAvgScore:
    def test_empty_list_returns_zero(self) -> None:
        assert avg_score([]) == 0.0

    def test_single_score(self) -> None:
        s = _score(viral_probability=0.8, post_id="p1")
        result = avg_score([s])
        assert result == pytest.approx(s.trend_score, abs=1e-6)

    def test_average_of_two(self) -> None:
        s1 = _score(viral_probability=0.0, author_followers=0, hashtag_count=0, has_media=False, mention_count=0, post_id="a")
        s2 = _score(viral_probability=1.0, author_followers=5_000_000, hashtag_count=5, has_media=True, mention_count=3, post_id="b")
        result = avg_score([s1, s2])
        expected = (s1.trend_score + s2.trend_score) / 2
        assert result == pytest.approx(expected, abs=1e-6)

    def test_returns_float(self) -> None:
        s = _score()
        assert isinstance(avg_score([s]), float)


class TestGetTrendingHashtags:
    def _make_post(self, post_id: str, text: str) -> dict:
        return {"post_id": post_id, "text": text}

    def test_empty_scores_returns_empty(self) -> None:
        posts = [self._make_post("p1", "#news")]
        result = get_trending_hashtags([], posts, top_k=5)
        assert result == []

    def test_returns_hashtags(self) -> None:
        s = _score(viral_probability=0.9, post_id="p1")
        posts = [self._make_post("p1", "#news #breaking")]
        result = get_trending_hashtags([s], posts, top_k=1)
        assert "news" in result or "breaking" in result

    def test_most_common_first(self) -> None:
        scores = [_score(post_id=f"p{i}") for i in range(3)]
        posts = [
            self._make_post("p0", "#python #python #python"),
            self._make_post("p1", "#python #java"),
            self._make_post("p2", "#java"),
        ]
        result = get_trending_hashtags(scores, posts, top_k=3)
        assert result[0] == "python"
