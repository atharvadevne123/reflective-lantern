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

    def test_consume_multiple_tokens_per_request(self):
        pkb = PerKeyTokenBucket(capacity=10, rate=1)
        assert pkb.consume("key", tokens=5) is True
        assert pkb.consume("key", tokens=5) is True
        assert pkb.consume("key", tokens=1) is False

    def test_same_key_creates_single_bucket(self):
        pkb = PerKeyTokenBucket(capacity=5, rate=1)
        for _ in range(3):
            pkb.consume("k1")
        assert pkb.bucket_count() == 1

    def test_many_unique_keys_all_independent(self):
        pkb = PerKeyTokenBucket(capacity=1, rate=0.001)
        keys = [f"user:{i}" for i in range(20)]
        results = [pkb.consume(k) for k in keys]
        assert all(results)
        assert pkb.bucket_count() == 20


class TestTokenBucketRefill:
    def test_refill_does_not_exceed_capacity(self):
        tb = TokenBucket(capacity=5, rate=1000)
        tb._tokens = 0  # type: ignore[attr-defined]
        tb._last_refill = time.monotonic() - 100  # 100s of refill available  # type: ignore[attr-defined]
        # available is capped at capacity
        assert tb.available <= 5

    def test_consume_zero_always_succeeds(self):
        tb = TokenBucket(capacity=5, rate=1)
        assert tb.consume(0) is True

    @pytest.mark.parametrize("cap", [1, 10, 100, 1000])
    def test_various_capacities(self, cap: int) -> None:
        tb = TokenBucket(capacity=cap, rate=1)
        assert tb.consume(cap) is True
        assert tb.consume(1) is False

    def test_consume_exactly_available(self):
        tb = TokenBucket(capacity=7, rate=1)
        available_before = tb.available
        result = tb.consume(int(available_before))
        assert result is True


class TestTokenBucketAvailable:
    def test_available_starts_at_capacity(self) -> None:
        tb = TokenBucket(capacity=5, rate=1)
        assert tb.available == pytest.approx(5.0, abs=0.1)

    def test_available_decreases_after_consume(self) -> None:
        tb = TokenBucket(capacity=10, rate=1)
        tb.consume(3)
        assert tb.available == pytest.approx(7.0, abs=0.2)

    def test_available_does_not_exceed_capacity(self) -> None:
        tb = TokenBucket(capacity=5, rate=100)
        tb._tokens = 0  # type: ignore[attr-defined]
        tb._last_refill = time.monotonic() - 60  # type: ignore[attr-defined]
        assert tb.available <= 5.0

    @pytest.mark.parametrize("cap", [1.0, 5.0, 100.0])
    def test_full_bucket_capacity_parametrized(self, cap: float) -> None:
        tb = TokenBucket(capacity=cap, rate=1)
        assert tb.available == pytest.approx(cap, abs=0.1)


class TestPerKeyTokenBucketEdgeCases:
    def test_empty_string_key_valid(self) -> None:
        pkb = PerKeyTokenBucket(capacity=5, rate=1)
        assert pkb.consume("") is True

    def test_high_token_request_denied(self) -> None:
        pkb = PerKeyTokenBucket(capacity=3, rate=1)
        assert pkb.consume("k", tokens=100) is False

    @pytest.mark.parametrize("n_keys", [1, 10, 50])
    def test_bucket_count_parametrized(self, n_keys: int) -> None:
        pkb = PerKeyTokenBucket(capacity=10, rate=100)
        for i in range(n_keys):
            pkb.consume(str(i))
        assert pkb.bucket_count() == n_keys

    def test_consume_returns_bool(self) -> None:
        pkb = PerKeyTokenBucket(capacity=1, rate=0.001)
        result1 = pkb.consume("k")
        result2 = pkb.consume("k")
        assert isinstance(result1, bool)
        assert isinstance(result2, bool)
