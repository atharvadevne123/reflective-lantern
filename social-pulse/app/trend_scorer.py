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
    composite = (
        probability_weight * viral_probability
        + (1.0 - probability_weight) * ep
    )
    composite = min(1.0, max(0.0, composite))
    tier = _tier_label(composite)
    logger.debug(
        "score_post post_id=%s vp=%.3f ep=%.3f composite=%.3f tier=%s",
        post_id, viral_probability, ep, composite, tier,
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
