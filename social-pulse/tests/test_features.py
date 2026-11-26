"""Tests for the NLP feature engineering pipeline."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.features import (
    TextStatTransformer,
    avg_word_length,
    build_feature_pipeline,
    capitalisation_ratio,
    char_ngram_density,
    count_emojis,
    count_hashtags,
    count_mentions,
    encode_cyclical,
    follower_bucket,
    has_url,
    hashtag_diversity_score,
    sentiment_score,
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
    @pytest.mark.parametrize(
        "followers,expected",
        [
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
        ],
    )
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
        assert len(self.transformer.get_feature_names_out()) == len(
            TextStatTransformer.FEATURE_NAMES
        )

    def test_no_nan_in_output(self) -> None:
        result = self.transformer.transform(self.sample)
        assert not np.any(np.isnan(result))


class TestBuildFeaturePipeline:
    def test_pipeline_transforms_samples(self) -> None:
        pipeline = build_feature_pipeline(tfidf_features=50)
        samples = [
            {
                "text": "hello world #foo",
                "author_followers": 100,
                "has_media": False,
                "has_link": False,
                "hashtag_count": 1,
                "mention_count": 0,
                "hour_of_day": 10,
                "day_of_week": 2,
            },
            {
                "text": "breaking news @alice #trend",
                "author_followers": 10000,
                "has_media": True,
                "has_link": True,
                "hashtag_count": 1,
                "mention_count": 1,
                "hour_of_day": 20,
                "day_of_week": 5,
            },
        ]
        result = pipeline.fit_transform(samples)
        assert result.shape[0] == 2
        assert result.shape[1] > 0


@pytest.mark.parametrize(
    "text,expected_min",
    [
        ("#a #b #c", 3),
        ("#one", 1),
        ("no tags", 0),
    ],
)
def test_count_hashtags_parametrized(text: str, expected_min: int) -> None:
    assert count_hashtags(text) >= expected_min


@pytest.mark.parametrize(
    "followers,expected_bucket",
    [
        (0, 0),
        (50, 0),
        (100, 1),
        (5000, 2),
        (50000, 3),
        (500000, 4),
        (5000000, 5),
        (50000000, 6),
    ],
)
def test_follower_bucket_parametrized(followers: int, expected_bucket: int) -> None:
    assert follower_bucket(followers) == expected_bucket


@pytest.mark.parametrize(
    "text,expected",
    [
        ("HELLO", 1.0),
        ("hello", 0.0),
        ("", 0.0),
    ],
)
def test_capitalisation_ratio_parametrized(text: str, expected: float) -> None:
    import math

    assert math.isclose(capitalisation_ratio(text), expected)


class TestCharNgramDensity:
    def test_too_short_returns_zero(self) -> None:
        assert char_ngram_density("ab", n=3) == 0.0

    def test_all_unique_ngrams(self) -> None:
        result = char_ngram_density("abcdef", n=3)
        assert result == pytest.approx(1.0)

    def test_repetitive_text_low_density(self) -> None:
        result = char_ngram_density("aaaaaa", n=3)
        assert result < 0.5

    def test_result_in_unit_interval(self) -> None:
        result = char_ngram_density("hello world today", n=3)
        assert 0.0 <= result <= 1.0

    def test_empty_string_returns_zero(self) -> None:
        assert char_ngram_density("", n=3) == 0.0

    @pytest.mark.parametrize(
        "text,n,expected_min",
        [
            ("abcde", 2, 0.5),
            ("aaaaa", 3, 0.0),
            ("hello world", 4, 0.1),
        ],
    )
    def test_parametrized(self, text: str, n: int, expected_min: float) -> None:
        result = char_ngram_density(text, n=n)
        assert result >= expected_min


class TestSentimentScore:
    def test_neutral_text_returns_zero(self) -> None:
        assert sentiment_score("some plain text") == 0.0

    def test_positive_text_returns_positive(self) -> None:
        assert sentiment_score("this is great and amazing") > 0.0

    def test_negative_text_returns_negative(self) -> None:
        assert sentiment_score("this is terrible and awful") < 0.0

    def test_result_in_minus_one_to_one(self) -> None:
        score = sentiment_score("love hate amazing bad best worst")
        assert -1.0 <= score <= 1.0

    def test_empty_string_returns_zero(self) -> None:
        assert sentiment_score("") == 0.0

    def test_mixed_sentiment(self) -> None:
        score = sentiment_score("love hate")
        assert score == 0.0  # equal pos and neg

    @pytest.mark.parametrize(
        "text,expected_sign",
        [
            ("great awesome amazing best", 1),
            ("terrible awful hate worst", -1),
            ("hello world today", 0),
        ],
    )
    def test_sentiment_direction(self, text: str, expected_sign: int) -> None:
        score = sentiment_score(text)
        if expected_sign > 0:
            assert score > 0
        elif expected_sign < 0:
            assert score < 0
        else:
            assert score == 0.0


class TestHashtagDiversityScore:
    def test_no_hashtags_returns_zero(self) -> None:
        assert hashtag_diversity_score("no tags here") == 0.0

    def test_all_unique_returns_one(self) -> None:
        assert hashtag_diversity_score("#a #b #c") == pytest.approx(1.0)

    def test_all_same_returns_one(self) -> None:
        assert hashtag_diversity_score("#same #same #same") == pytest.approx(1.0 / 3.0)

    def test_mixed_returns_fraction(self) -> None:
        score = hashtag_diversity_score("#a #a #b")
        assert 0.0 < score < 1.0

    def test_result_in_unit_interval(self) -> None:
        score = hashtag_diversity_score("#x #y #x #z")
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("#a #b #c #d", 1.0),
            ("#a #a #a #a", 0.25),
            ("no tags", 0.0),
        ],
    )
    def test_parametrized(self, text: str, expected: float) -> None:
        assert hashtag_diversity_score(text) == pytest.approx(expected, abs=0.01)
