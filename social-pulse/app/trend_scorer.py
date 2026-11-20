"""Trend scoring utilities for ranking posts by predicted viral potential."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrendScore:
    """Composite trend score for a single post.

    Combines the model's viral probability with content-quality signals
    to produce a single sortable score.

    Attributes:
        post_id: Caller-supplied identifier for the post.
        viral_probability: Raw ensemble probability (0–1).
        engagement_potential: Weighted blend of follower reach and content signals.
        trend_score: Final composite score (higher = more likely to trend).
        tier: Human-readable tier label based on trend_score.
    """

    post_id: str
    viral_probability: float
    engagement_potential: float
    trend_score: float
    tier: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "post_id": self.post_id,
            "viral_probability": round(self.viral_probability, 4),
            "engagement_potential": round(self.engagement_potential, 4),
            "trend_score": round(self.trend_score, 4),
            "tier": self.tier,
        }


def _engagement_potential(
    author_followers: int,
    hashtag_count: int,
    has_media: bool,
    mention_count: int,
) -> float:
    """Compute a normalised engagement-potential signal (0–1).

    Uses log-follower reach, hashtag diversity bonus, and media presence.

    Args:
        author_followers: Author follower count.
        hashtag_count: Number of hashtags.
        has_media: Whether the post contains media.
        mention_count: Number of @mentions.

    Returns:
        Float in [0, 1].
    """
    import math

    follower_signal = min(1.0, math.log1p(author_followers) / 20.0)
    hashtag_signal = min(1.0, hashtag_count / 5.0) * 0.3
    media_bonus = 0.2 if has_media else 0.0
    mention_signal = min(1.0, mention_count / 3.0) * 0.1
    raw = follower_signal * 0.4 + hashtag_signal + media_bonus + mention_signal
    return min(1.0, raw)


def _tier_label(score: float) -> str:
    """Map a trend score to a human-readable tier label.

    Args:
        score: Composite trend score in [0, 1].

    Returns:
        One of "Viral", "High", "Medium", or "Low".
    """
    if score >= 0.75:
        return "Viral"
    if score >= 0.50:
        return "High"
    if score >= 0.25:
        return "Medium"
    return "Low"


def score_post(
    post_id: str,
    viral_probability: float,
    author_followers: int,
    hashtag_count: int,
    has_media: bool,
    mention_count: int,
    probability_weight: float = 0.6,
) -> TrendScore:
    """Compute a composite trend score for a single post.

    Args:
        post_id: Unique identifier for the post.
        viral_probability: Model output probability (0–1).
        author_followers: Author follower count.
        hashtag_count: Number of hashtags.
        has_media: Whether the post contains media.
        mention_count: Number of @mentions.
        probability_weight: Weight given to viral_probability vs. engagement_potential.

    Returns:
        A populated TrendScore instance.
    """
    ep = _engagement_potential(author_followers, hashtag_count, has_media, mention_count)
    composite = probability_weight * viral_probability + (1.0 - probability_weight) * ep
    composite = min(1.0, max(0.0, composite))
    tier = _tier_label(composite)
    logger.debug(
        "score_post post_id=%s vp=%.3f ep=%.3f composite=%.3f tier=%s",
        post_id,
        viral_probability,
        ep,
        composite,
        tier,
    )
    return TrendScore(
        post_id=post_id,
        viral_probability=viral_probability,
        engagement_potential=ep,
        trend_score=composite,
        tier=tier,
    )


def rank_posts(scores: list[TrendScore]) -> list[TrendScore]:
    """Sort a list of TrendScores from highest to lowest trend_score.

    Args:
        scores: List of TrendScore instances to rank.

    Returns:
        A new list sorted by trend_score descending.
    """
    return sorted(scores, key=lambda s: s.trend_score, reverse=True)


def top_n_posts(scores: list[TrendScore], n: int = 10) -> list[TrendScore]:
    """Return at most *n* highest-scoring posts.

    Args:
        scores: Unordered list of TrendScore instances.
        n: Maximum number of results to return.

    Returns:
        Up to *n* TrendScore instances sorted by trend_score descending.
    """
    return rank_posts(scores)[:n]


def filter_by_tier(scores: list[TrendScore], tier: str) -> list[TrendScore]:
    """Return only the scores whose tier matches *tier* (case-insensitive).

    Args:
        scores: List of TrendScore instances.
        tier: One of "Viral", "High", "Medium", or "Low".

    Returns:
        Filtered list preserving original order.
    """
    target = tier.strip().capitalize()
    return [s for s in scores if s.tier == target]


def avg_score(scores: list[TrendScore]) -> float:
    """Return the mean trend_score across all *scores*.

    Args:
        scores: List of TrendScore instances.

    Returns:
        Mean trend_score, or 0.0 if the list is empty.
    """
    if not scores:
        return 0.0
    return sum(s.trend_score for s in scores) / len(scores)


def get_trending_hashtags(scores: list[TrendScore], posts: list[dict[str, Any]], top_k: int = 5) -> list[str]:
    """Return the most common hashtags among the top-scoring posts.

    Args:
        scores: Ranked list of TrendScore instances.
        posts: Original post dicts keyed by "post_id" and "text".
        top_k: Number of top-k posts to scan for hashtags.

    Returns:
        List of hashtag strings (without #) sorted by frequency descending.
    """
    import re
    from collections import Counter

    top_ids = {s.post_id for s in scores[:top_k]}
    counter: Counter[str] = Counter()
    for post in posts:
        if post.get("post_id") in top_ids:
            tags = re.findall(r"#(\w+)", post.get("text", "").lower())
            counter.update(tags)
    return [tag for tag, _ in counter.most_common()]
