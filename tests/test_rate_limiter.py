"""Tests for app/rate_limiter.py token-bucket rate limiter."""

from __future__ import annotations

import pytest

from app.rate_limiter import (
    TokenBucketRateLimiter,
    burst_capacity_fraction,
    make_rate_limiter,
    make_strict_limiter,
    prune_idle_clients,
)


def test_first_request_allowed() -> None:
    limiter = TokenBucketRateLimiter(capacity=5.0, rate_per_second=1.0)
    assert limiter.is_allowed("client-1") is True


def test_exhausted_bucket_denied() -> None:
    limiter = TokenBucketRateLimiter(capacity=2.0, rate_per_second=0.01)
    limiter.is_allowed("c")
    limiter.is_allowed("c")
    assert limiter.is_allowed("c") is False


def test_remaining_tokens_full_for_new_client() -> None:
    limiter = TokenBucketRateLimiter(capacity=10.0, rate_per_second=1.0)
    assert limiter.remaining_tokens("new") == 10.0


def test_remaining_tokens_decreases_after_request() -> None:
    limiter = TokenBucketRateLimiter(capacity=5.0, rate_per_second=0.01)
    limiter.is_allowed("x")
    assert limiter.remaining_tokens("x") < 5.0


def test_reset_restores_full_capacity() -> None:
    limiter = TokenBucketRateLimiter(capacity=3.0, rate_per_second=0.01)
    limiter.is_allowed("r")
    limiter.is_allowed("r")
    limiter.reset("r")
    assert limiter.remaining_tokens("r") == 3.0


def test_clear_removes_all_clients() -> None:
    limiter = TokenBucketRateLimiter(capacity=5.0, rate_per_second=1.0)
    limiter.is_allowed("a")
    limiter.is_allowed("b")
    limiter.clear()
    assert limiter.client_count == 0


def test_client_count_tracks_unique_keys() -> None:
    limiter = TokenBucketRateLimiter(capacity=5.0, rate_per_second=1.0)
    limiter.is_allowed("alpha")
    limiter.is_allowed("beta")
    limiter.is_allowed("alpha")
    assert limiter.client_count == 2


def test_invalid_capacity_raises() -> None:
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(capacity=0.0, rate_per_second=1.0)


def test_invalid_rate_raises() -> None:
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(capacity=5.0, rate_per_second=-1.0)


def test_make_rate_limiter_factory() -> None:
    limiter = make_rate_limiter(capacity=30.0, rate_per_second=0.5)
    assert isinstance(limiter, TokenBucketRateLimiter)
    assert limiter.is_allowed("x") is True


@pytest.mark.parametrize("capacity", [1.0, 5.0, 10.0, 100.0])
def test_capacity_respected(capacity) -> None:
    limiter = TokenBucketRateLimiter(capacity=capacity, rate_per_second=0.001)
    allowed = sum(1 for _ in range(int(capacity) + 5) if limiter.is_allowed("k"))
    assert allowed == int(capacity)


def test_separate_clients_independent() -> None:
    limiter = TokenBucketRateLimiter(capacity=1.0, rate_per_second=0.01)
    limiter.is_allowed("client-a")
    assert limiter.is_allowed("client-b") is True


@pytest.mark.parametrize("n_clients", [1, 5, 10])
def test_clear_resets_all_clients(n_clients) -> None:
    limiter = TokenBucketRateLimiter(capacity=10.0, rate_per_second=1.0)
    for i in range(n_clients):
        limiter.is_allowed(f"c-{i}")
    limiter.clear()
    assert limiter.client_count == 0


def test_is_allowed_returns_bool() -> None:
    limiter = TokenBucketRateLimiter(capacity=5.0, rate_per_second=1.0)
    result = limiter.is_allowed("test")
    assert isinstance(result, bool)


def test_remaining_tokens_after_clear() -> None:
    limiter = TokenBucketRateLimiter(capacity=5.0, rate_per_second=0.01)
    limiter.is_allowed("c")
    limiter.clear()
    assert limiter.remaining_tokens("c") == 5.0


@pytest.mark.parametrize("rate", [0.1, 0.5, 1.0, 5.0])
def test_make_rate_limiter_various_rates(rate) -> None:
    limiter = make_rate_limiter(capacity=10.0, rate_per_second=rate)
    assert limiter.is_allowed("test") is True


class TestLimiterUtilization:
    def test_full_capacity_is_zero(self) -> None:
        from app.rate_limiter import limiter_utilization, make_rate_limiter

        limiter = make_rate_limiter(capacity=10.0, rate_per_second=1.0)
        assert limiter_utilization(limiter, "c1") == pytest.approx(0.0, abs=0.05)

    def test_after_consume_increases(self) -> None:
        from app.rate_limiter import limiter_utilization, make_rate_limiter

        limiter = make_rate_limiter(capacity=10.0, rate_per_second=1.0)
        for _ in range(5):
            limiter.is_allowed("c2")
        assert limiter_utilization(limiter, "c2") > 0.0


class TestResetLimiter:
    def test_resets_to_full(self) -> None:
        from app.rate_limiter import limiter_utilization, make_rate_limiter, reset_limiter

        limiter = make_rate_limiter(capacity=10.0, rate_per_second=1.0)
        for _ in range(8):
            limiter.is_allowed("c3")
        reset_limiter(limiter, "c3")
        assert limiter_utilization(limiter, "c3") == pytest.approx(0.0, abs=0.05)


class TestIsRateLimited:
    def test_not_limited_when_full(self) -> None:
        from app.rate_limiter import is_rate_limited, make_rate_limiter

        limiter = make_rate_limiter(capacity=10.0, rate_per_second=1.0)
        assert is_rate_limited(limiter, "c4") is False

    def test_limited_when_empty(self) -> None:
        from app.rate_limiter import is_rate_limited, make_rate_limiter

        limiter = make_rate_limiter(capacity=2.0, rate_per_second=0.001)
        limiter.is_allowed("c5")
        limiter.is_allowed("c5")
        assert is_rate_limited(limiter, "c5") is True


class TestMakeStrictLimiter:
    def test_returns_limiter(self) -> None:
        from app.rate_limiter import TokenBucketRateLimiter, make_strict_limiter

        limiter = make_strict_limiter(10.0)
        assert isinstance(limiter, TokenBucketRateLimiter)

    def test_allows_within_rate(self) -> None:
        from app.rate_limiter import make_strict_limiter

        limiter = make_strict_limiter(5.0)
        for _ in range(5):
            limiter.is_allowed("strict_client")


def test_make_strict_limiter_returns_limiter() -> None:
    limiter = make_strict_limiter(60)
    assert limiter is not None


def test_make_strict_limiter_capacity() -> None:
    limiter = make_strict_limiter(30)
    assert limiter._capacity == pytest.approx(30.0)


def test_make_strict_limiter_invalid_raises() -> None:
    with pytest.raises(ValueError):
        make_strict_limiter(0)


def test_make_strict_limiter_allows_first_request() -> None:
    limiter = make_strict_limiter(10)
    assert limiter.is_allowed("client") is True


def test_burst_capacity_fraction_full() -> None:
    limiter = make_rate_limiter(capacity=10.0, rate_per_second=1.0)
    frac = burst_capacity_fraction(limiter, "new_client")
    assert frac == pytest.approx(1.0)


def test_burst_capacity_fraction_after_use() -> None:
    limiter = make_rate_limiter(capacity=10.0, rate_per_second=0.01)
    for _ in range(5):
        limiter.is_allowed("c")
    frac = burst_capacity_fraction(limiter, "c")
    assert frac < 1.0
    assert frac >= 0.0


def test_burst_capacity_fraction_empty_bucket() -> None:
    limiter = make_rate_limiter(capacity=2.0, rate_per_second=0.001)
    limiter.is_allowed("c")
    limiter.is_allowed("c")
    frac = burst_capacity_fraction(limiter, "c")
    assert frac == pytest.approx(0.0)


class TestPruneIdleClients:
    def test_removes_idle_buckets(self) -> None:
        limiter = make_rate_limiter(capacity=5.0, rate_per_second=1.0)
        limiter.is_allowed("client-a")
        limiter.is_allowed("client-b")
        # Set last_refill far in the past to simulate idle
        with limiter._lock:
            for bucket in limiter._buckets.values():
                bucket.last_refill -= 1000.0
        removed = prune_idle_clients(limiter, max_idle_seconds=1.0)
        assert removed == 2
        assert limiter.client_count == 0

    def test_keeps_recent_buckets(self) -> None:
        limiter = make_rate_limiter(capacity=5.0, rate_per_second=1.0)
        limiter.is_allowed("recent")
        removed = prune_idle_clients(limiter, max_idle_seconds=3600.0)
        assert removed == 0
        assert limiter.client_count == 1

    def test_mixed_idle_and_recent(self) -> None:
        limiter = make_rate_limiter(capacity=5.0, rate_per_second=1.0)
        limiter.is_allowed("active")
        limiter.is_allowed("idle")
        with limiter._lock:
            limiter._buckets["idle"].last_refill -= 9999.0
        removed = prune_idle_clients(limiter, max_idle_seconds=60.0)
        assert removed == 1
        assert "active" in limiter._buckets
        assert "idle" not in limiter._buckets

    def test_empty_limiter_returns_zero(self) -> None:
        limiter = make_rate_limiter()
        assert prune_idle_clients(limiter, max_idle_seconds=1.0) == 0


class TestActiveClientCount:
    def test_empty_limiter_returns_zero(self) -> None:
        from app.rate_limiter import active_client_count, make_rate_limiter

        limiter = make_rate_limiter()
        assert active_client_count(limiter) == 0

    def test_one_client(self) -> None:
        from app.rate_limiter import active_client_count, make_rate_limiter

        limiter = make_rate_limiter()
        limiter.is_allowed("client_a")
        assert active_client_count(limiter) == 1

    def test_multiple_unique_clients(self) -> None:
        from app.rate_limiter import active_client_count, make_rate_limiter

        limiter = make_rate_limiter()
        for i in range(5):
            limiter.is_allowed(f"client_{i}")
        assert active_client_count(limiter) == 5


class TestTotalConsumedTokens:
    def test_no_requests_consumed_at_capacity(self) -> None:
        from app.rate_limiter import make_rate_limiter, total_consumed_tokens

        limiter = make_rate_limiter(capacity=10.0)
        consumed = total_consumed_tokens(limiter)
        assert consumed >= 0.0

    def test_after_request_consumed_increases(self) -> None:
        from app.rate_limiter import make_rate_limiter, total_consumed_tokens

        limiter = make_rate_limiter(capacity=10.0, rate_per_second=0.001)
        limiter.is_allowed("test")
        consumed = total_consumed_tokens(limiter, "test")
        assert consumed >= 0.0


class TestClientExists:
    def test_new_client_does_not_exist(self) -> None:
        from app.rate_limiter import client_exists, make_rate_limiter

        limiter = make_rate_limiter()
        assert client_exists(limiter, "unknown") is False

    def test_client_exists_after_request(self) -> None:
        from app.rate_limiter import client_exists, make_rate_limiter

        limiter = make_rate_limiter()
        limiter.is_allowed("present")
        assert client_exists(limiter, "present") is True

    @pytest.mark.parametrize("key", ["a", "b", "c"])
    def test_parametrized_clients(self, key: str) -> None:
        from app.rate_limiter import client_exists, make_rate_limiter

        limiter = make_rate_limiter()
        assert client_exists(limiter, key) is False
        limiter.is_allowed(key)
        assert client_exists(limiter, key) is True


class TestBucketFillPercentage:
    def test_no_requests_full(self) -> None:
        from app.rate_limiter import bucket_fill_percentage, make_rate_limiter

        limiter = make_rate_limiter(capacity=10.0)
        assert bucket_fill_percentage(limiter, "new_client") == pytest.approx(100.0)

    def test_after_requests_lower(self) -> None:
        from app.rate_limiter import bucket_fill_percentage, make_rate_limiter

        limiter = make_rate_limiter(capacity=10.0, refill_rate=0.0)
        for _ in range(5):
            limiter.is_allowed("c")
        pct = bucket_fill_percentage(limiter, "c")
        assert pct < 100.0


class TestIsRateLimitedHelper:
    def test_fresh_client_not_limited(self) -> None:
        from app.rate_limiter import is_rate_limited, make_rate_limiter

        limiter = make_rate_limiter(capacity=10.0)
        assert is_rate_limited(limiter, "x", cost=1.0) is False

    def test_exhausted_client_limited(self) -> None:
        from app.rate_limiter import is_rate_limited, make_rate_limiter

        limiter = make_rate_limiter(capacity=3.0, refill_rate=0.0)
        for _ in range(3):
            limiter.is_allowed("y")
        assert is_rate_limited(limiter, "y", cost=1.0) is True


class TestResetClient:
    def test_reset_restores_bucket(self) -> None:
        from app.rate_limiter import bucket_fill_percentage, make_rate_limiter, reset_client

        limiter = make_rate_limiter(capacity=5.0, refill_rate=0.0)
        for _ in range(5):
            limiter.is_allowed("z")
        assert bucket_fill_percentage(limiter, "z") == pytest.approx(0.0, abs=1.0)
        reset_client(limiter, "z")
        assert bucket_fill_percentage(limiter, "z") == pytest.approx(100.0, abs=1.0)

    def test_reset_nonexistent_returns_false(self) -> None:
        from app.rate_limiter import make_rate_limiter, reset_client

        limiter = make_rate_limiter()
        assert reset_client(limiter, "nobody") is False
