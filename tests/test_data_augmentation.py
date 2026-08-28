"""Tests for app.data_augmentation."""

import random
import re

from app.data_augmentation import (
    AugmentationConfig,
    augment_batch,
    augment_text,
    jitter_numerics,
    random_deletion,
    random_swap,
    synonym_replace,
)


def make_rng(seed: int = 42) -> random.Random:
    return random.Random(seed)


class TestSynonymReplace:
    def test_replaces_known_token(self):
        rng = make_rng(0)
        synonyms = {"quick": ["fast"]}
        # force replace by patching prob to 1
        result = synonym_replace(["quick"], synonyms, prob=1.0, rng=rng)
        assert result == ["fast"]

    def test_keeps_unknown_token(self):
        rng = make_rng(0)
        result = synonym_replace(["hello"], {}, prob=1.0, rng=rng)
        assert result == ["hello"]

    def test_zero_prob_no_change(self):
        rng = make_rng(0)
        synonyms = {"a": ["b"]}
        result = synonym_replace(["a", "a"], synonyms, prob=0.0, rng=rng)
        assert result == ["a", "a"]


class TestRandomDeletion:
    def test_deletes_some_tokens(self):
        rng = make_rng(42)
        tokens = ["a", "b", "c", "d", "e"]
        result = random_deletion(tokens, prob=0.9, rng=rng)
        assert len(result) >= 1

    def test_single_token_preserved(self):
        rng = make_rng(0)
        result = random_deletion(["only"], prob=1.0, rng=rng)
        assert result == ["only"]

    def test_zero_prob_no_deletion(self):
        rng = make_rng(0)
        tokens = ["x", "y", "z"]
        result = random_deletion(tokens, prob=0.0, rng=rng)
        assert result == tokens


class TestRandomSwap:
    def test_length_preserved(self):
        rng = make_rng(7)
        tokens = ["a", "b", "c", "d"]
        result = random_swap(tokens, prob=1.0, rng=rng)
        assert len(result) == 4
        assert sorted(result) == sorted(tokens)

    def test_zero_prob_no_swap(self):
        rng = make_rng(0)
        tokens = ["x", "y"]
        result = random_swap(tokens, prob=0.0, rng=rng)
        assert result == ["x", "y"]


class TestJitterNumerics:
    def test_numeric_changed(self):
        rng = make_rng(1)
        text = "value is 100"
        out = jitter_numerics(text, pct=0.1, rng=rng)
        nums = re.findall(r"-?\d+(?:\.\d+)?", out)
        assert float(nums[-1]) != 100.0

    def test_zero_value_jitter(self):
        rng = make_rng(1)
        out = jitter_numerics("zero is 0", pct=0.05, rng=rng)
        # should not raise, output still contains a number
        assert re.search(r"-?\d+(?:\.\d+)?", out) is not None

    def test_no_numbers_unchanged(self):
        rng = make_rng(0)
        out = jitter_numerics("no numbers here", pct=0.1, rng=rng)
        assert out == "no numbers here"


class TestAugmentText:
    def test_returns_string(self):
        cfg = AugmentationConfig(seed=0)
        out = augment_text("hello world", cfg)
        assert isinstance(out, str)
        assert len(out) > 0

    def test_synonym_applied(self):
        cfg = AugmentationConfig(
            synonym_prob=1.0,
            deletion_prob=0.0,
            swap_prob=0.0,
            numeric_jitter_pct=0.0,
            seed=0,
            synonyms={"hello": ["hi"], "world": ["earth"]},
        )
        out = augment_text("hello world", cfg)
        assert "hi" in out or "earth" in out


class TestAugmentBatch:
    def test_output_size(self):
        cfg = AugmentationConfig(seed=0)
        samples = ["a b", "c d", "e f"]
        out = augment_batch(samples, cfg, n_augments=2)
        assert len(out) == 6

    def test_custom_transform(self):
        cfg = AugmentationConfig(seed=0)
        out = augment_batch(["hello"], cfg, n_augments=1, transform=lambda t, c: t.upper())
        assert out == ["HELLO"]
