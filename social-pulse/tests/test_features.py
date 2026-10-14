"""Tests for the NLP feature engineering pipeline."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.features import (
    TextStatTransformer,
    avg_word_length,
    capitalisation_ratio,
    count_emojis,
    count_hashtags,
    count_mentions,
    encode_cyclical,
    exclamation_density,
    follower_bucket,
    has_url,
    question_density,
    build_feature_pipeline,
)


class TestCountHashtags:
    def test_counts_single_hashtag(self) -> None:
        assert count_hashtags("hello #world") == 1

    def test_counts_multiple_hashtags(self) -> None:
        assert count_hashtags("#a #b #c") == 3

    def test_no_hashtags(self) -> None:
        assert count_hashtags("no tags here") == 0

    def test_empty_string(self) -> None:
        assert count_hashtags("") == 0


class TestCountMentions:
    def test_counts_single_mention(self) -> None:
        assert count_mentions("hi @alice") == 1

    def test_counts_multiple_mentions(self) -> None:
        assert count_mentions("@a @b @c") == 3

    def test_no_mentions(self) -> None:
        assert count_mentions("no mentions") == 0


class TestCountEmojis:
    def test_counts_emoji(self) -> None:
        assert count_emojis("hello 😀 world") >= 1

    def test_no_emoji(self) -> None:
        assert count_emojis("plain text") == 0


class TestHasUrl:
    def test_detects_http(self) -> None:
        assert has_url("check https://example.com") == 1

    def test_detects_www(self) -> None:
        assert has_url("visit www.example.com") == 1

    def test_no_url(self) -> None:
        assert has_url("no link here") == 0


class TestEncodeCyclical:
    def test_zero_encodes_correctly(self) -> None:
        s, c = encode_cyclical(0, 24)
        assert math.isclose(s, 0.0, abs_tol=1e-6)
        assert math.isclose(c, 1.0, abs_tol=1e-6)

    def test_quarter_period_sin_is_one(self) -> None:
        s, c = encode_cyclical(6, 24)
        assert math.isclose(s, 1.0, abs_tol=1e-6)
        assert math.isclose(c, 0.0, abs_tol=1e-6)

    def test_half_period_cos_is_minus_one(self) -> None:
        _, c = encode_cyclical(12, 24)
        assert math.isclose(c, -1.0, abs_tol=1e-6)


class TestFollowerBucket:
    @pytest.mark.parametrize("followers,expected", [
        (0, 0),
        (99, 0),
        (100, 1),
        (999, 1),
        (1000, 2),
        (9999, 2),
        (10000, 3),
        (99999, 3),
        (100000, 4),
        (999999, 4),
        (1000000, 5),
        (9999999, 5),
        (10000000, 6),
    ])
    def test_bucket_boundaries(self, followers: int, expected: int) -> None:
        assert follower_bucket(followers) == expected


class TestCapitalisationRatio:
    def test_all_uppercase(self) -> None:
        assert math.isclose(capitalisation_ratio("HELLO"), 1.0)

    def test_all_lowercase(self) -> None:
        assert math.isclose(capitalisation_ratio("hello"), 0.0)

    def test_mixed(self) -> None:
        ratio = capitalisation_ratio("Hello")
        assert 0.0 < ratio < 1.0

    def test_empty_string(self) -> None:
        assert capitalisation_ratio("") == 0.0


class TestAvgWordLength:
    def test_single_word(self) -> None:
        assert math.isclose(avg_word_length("hello"), 5.0)

    def test_empty_string(self) -> None:
        assert avg_word_length("") == 0.0

    def test_strips_punctuation(self) -> None:
        result = avg_word_length("hello, world!")
        assert result > 0.0


class TestTextStatTransformer:
    def setup_method(self) -> None:
        self.transformer = TextStatTransformer()
        self.sample = [
            {
                "text": "#trending hello @user https://t.co/abc 😀!?",
                "author_followers": 50000,
                "has_media": True,
                "has_link": True,
                "hashtag_count": 1,
                "mention_count": 1,
                "hour_of_day": 18,
                "day_of_week": 4,
            }
        ]

    def test_fit_returns_self(self) -> None:
        assert self.transformer.fit(self.sample) is self.transformer

    def test_transform_returns_array(self) -> None:
        result = self.transformer.transform(self.sample)
        assert isinstance(result, np.ndarray)

    def test_shape_correct(self) -> None:
        result = self.transformer.transform(self.sample)
        assert result.shape == (1, len(TextStatTransformer.FEATURE_NAMES))

    def test_feature_names_length(self) -> None:
        assert len(self.transformer.get_feature_names_out()) == len(TextStatTransformer.FEATURE_NAMES)

    def test_no_nan_in_output(self) -> None:
        result = self.transformer.transform(self.sample)
        assert not np.any(np.isnan(result))


class TestBuildFeaturePipeline:
    def test_pipeline_transforms_samples(self) -> None:
        pipeline = build_feature_pipeline(tfidf_features=50)
        samples = [
            {"text": "hello world #foo", "author_followers": 100,
             "has_media": False, "has_link": False,
             "hashtag_count": 1, "mention_count": 0,
             "hour_of_day": 10, "day_of_week": 2},
            {"text": "breaking news @alice #trend", "author_followers": 10000,
             "has_media": True, "has_link": True,
             "hashtag_count": 1, "mention_count": 1,
             "hour_of_day": 20, "day_of_week": 5},
        ]
        result = pipeline.fit_transform(samples)
        assert result.shape[0] == 2
        assert result.shape[1] > 0


@pytest.mark.parametrize("text,expected_min", [
    ("#a #b #c", 3),
    ("#one", 1),
    ("no tags", 0),
])
def test_count_hashtags_parametrized(text: str, expected_min: int) -> None:
    assert count_hashtags(text) >= expected_min
