"""Tests for Pydantic schema validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import BatchPredictRequest, PostFeaturesFull


def _valid_post(**overrides) -> dict:
    base = {
        "text": "Check this out! #launch #product",
        "author_followers": 10000,
        "has_media": True,
        "has_link": False,
        "hashtag_count": 2,
        "mention_count": 0,
        "hour_of_day": 15,
        "day_of_week": 3,
    }
    base.update(overrides)
    return base


class TestPostFeaturesFull:
    def test_valid_post_parses(self) -> None:
        post = PostFeaturesFull(**_valid_post())
        assert post.text == "Check this out! #launch #product"

    def test_blank_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            PostFeaturesFull(**_valid_post(text="   "))

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            PostFeaturesFull(**_valid_post(text=""))

    def test_negative_followers_raises(self) -> None:
        with pytest.raises(ValidationError):
            PostFeaturesFull(**_valid_post(author_followers=-1))

    def test_excessive_followers_raises(self) -> None:
        with pytest.raises(ValidationError):
            PostFeaturesFull(**_valid_post(author_followers=600_000_000))

    def test_hour_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            PostFeaturesFull(**_valid_post(hour_of_day=24))

    def test_negative_hour_raises(self) -> None:
        with pytest.raises(ValidationError):
            PostFeaturesFull(**_valid_post(hour_of_day=-1))

    def test_dow_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            PostFeaturesFull(**_valid_post(day_of_week=7))

    def test_excessive_engagement_raises(self) -> None:
        with pytest.raises(ValidationError):
            PostFeaturesFull(**_valid_post(author_avg_engagement=20_000_000.0))

    def test_text_stripped(self) -> None:
        post = PostFeaturesFull(**_valid_post(text="  hello world  "))
        assert post.text == "hello world"

    def test_defaults_applied(self) -> None:
        post = PostFeaturesFull(**_valid_post())
        assert post.has_media is True
        assert post.has_link is False
        assert post.is_reply is False
        assert post.is_quote is False


class TestBatchPredictRequest:
    def test_single_post_valid(self) -> None:
        req = BatchPredictRequest(posts=[PostFeaturesFull(**_valid_post())])
        assert len(req.posts) == 1

    def test_empty_posts_raises(self) -> None:
        with pytest.raises(ValidationError):
            BatchPredictRequest(posts=[])

    def test_50_posts_valid(self) -> None:
        posts = [PostFeaturesFull(**_valid_post()) for _ in range(50)]
        req = BatchPredictRequest(posts=posts)
        assert len(req.posts) == 50

    def test_51_posts_raises(self) -> None:
        posts = [PostFeaturesFull(**_valid_post()) for _ in range(51)]
        with pytest.raises(ValidationError):
            BatchPredictRequest(posts=posts)


@pytest.mark.parametrize("hour", [0, 12, 23])
def test_valid_hours(hour: int) -> None:
    post = PostFeaturesFull(**_valid_post(hour_of_day=hour))
    assert post.hour_of_day == hour


@pytest.mark.parametrize("dow", [0, 3, 6])
def test_valid_days(dow: int) -> None:
    post = PostFeaturesFull(**_valid_post(day_of_week=dow))
    assert post.day_of_week == dow
