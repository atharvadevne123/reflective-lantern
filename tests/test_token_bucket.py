"""Tests for app.token_bucket module."""

from __future__ import annotations

import threading
import time

import pytest

from app.token_bucket import PerKeyTokenBucket, TokenBucket


class TestTokenBucket:
    def test_full_bucket_at_creation(self):
        tb = TokenBucket(capacity=10, rate=1)
        assert tb.available == pytest.approx(10, abs=0.1)

    def test_consume_within_capacity_succeeds(self):
        tb = TokenBucket(capacity=5, rate=1)
        assert tb.consume(3) is True

    def test_consume_exceeding_capacity_fails(self):
        tb = TokenBucket(capacity=5, rate=1)
        assert tb.consume(6) is False

    def test_tokens_decrease_after_consume(self):
        tb = TokenBucket(capacity=10, rate=1)
        tb.consume(4)
        assert tb.available == pytest.approx(6, abs=0.2)

    def test_refill_over_time(self, monkeypatch):
        tb = TokenBucket(capacity=10, rate=5)
        tb._tokens = 0
        tb._last_refill = time.monotonic() - 2  # 2s ago
        assert tb.available >= 10  # capped at capacity

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError, match="capacity"):
            TokenBucket(capacity=0, rate=1)

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError, match="rate"):
            TokenBucket(capacity=5, rate=-1)

    @pytest.mark.parametrize("tokens", [1, 2, 5])
    def test_exact_capacity_consume(self, tokens):
        tb = TokenBucket(capacity=tokens, rate=100)
        assert tb.consume(tokens) is True
        assert tb.consume(1) is False

    def test_thread_safety(self):
        tb = TokenBucket(capacity=100, rate=1000)
        results = []

        def worker():
            results.append(tb.consume(1))

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        success = sum(results)
        assert success <= 100

    def test_wait_and_consume_success(self):
        tb = TokenBucket(capacity=10, rate=100)
        assert tb.wait_and_consume(1, timeout=1) is True

    def test_wait_and_consume_timeout(self):
        tb = TokenBucket(capacity=1, rate=0.001)
        tb.consume(1)  # drain
        result = tb.wait_and_consume(1, timeout=0.05)
        assert result is False


class TestPerKeyTokenBucket:
    def test_consume_different_keys_are_independent(self):
        pkb = PerKeyTokenBucket(capacity=1, rate=100)
        assert pkb.consume("a") is True
        assert pkb.consume("b") is True  # fresh bucket

    def test_exhausted_key_fails(self):
        pkb = PerKeyTokenBucket(capacity=1, rate=0.001)
        pkb.consume("x")
        assert pkb.consume("x") is False

    def test_bucket_count_tracks_unique_keys(self):
        pkb = PerKeyTokenBucket(capacity=5, rate=1)
        pkb.consume("k1")
        pkb.consume("k2")
        assert pkb.bucket_count() == 2


class TestTokenBucketEdgeCases:
    def test_consume_zero_tokens_always_succeeds(self):
        from app.token_bucket import TokenBucket

        b = TokenBucket(capacity=1.0, rate=1.0)
        b.consume(1.0)  # empty bucket
        assert b.consume(0.0) is True

    def test_consume_more_than_capacity_fails(self):
        from app.token_bucket import TokenBucket

        b = TokenBucket(capacity=5.0, rate=1.0)
        assert b.consume(6.0) is False

    @pytest.mark.parametrize("cap", [1.0, 5.0, 10.0])
    def test_initial_tokens_equal_capacity(self, cap: float):
        from app.token_bucket import TokenBucket

        b = TokenBucket(capacity=cap, rate=1.0)
        # Consume all tokens
        for _ in range(int(cap)):
            b.consume(1.0)
        # Now bucket should be empty
        assert b.consume(1.0) is False

    def test_per_key_creates_independent_buckets(self):
        from app.token_bucket import PerKeyTokenBucket

        pkb = PerKeyTokenBucket(capacity=1.0, rate=1.0)
        assert pkb.consume("a") is True
        assert pkb.consume("a") is False
        assert pkb.consume("b") is True

    def test_per_key_bucket_count(self):
        from app.token_bucket import PerKeyTokenBucket

        pkb = PerKeyTokenBucket(capacity=5.0, rate=5.0)
        pkb.consume("x")
        pkb.consume("y")
        pkb.consume("z")
        assert pkb.bucket_count() == 3
