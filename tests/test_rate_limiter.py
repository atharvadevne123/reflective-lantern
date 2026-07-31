"""Tests for app/rate_limiter.py token-bucket rate limiter."""

from __future__ import annotations

import pytest

from app.rate_limiter import TokenBucketRateLimiter, make_rate_limiter


def test_first_request_allowed():
    limiter = TokenBucketRateLimiter(capacity=5.0, rate_per_second=1.0)
    assert limiter.is_allowed("client-1") is True


def test_exhausted_bucket_denied():
    limiter = TokenBucketRateLimiter(capacity=2.0, rate_per_second=0.01)
    limiter.is_allowed("c")
    limiter.is_allowed("c")
    assert limiter.is_allowed("c") is False


def test_remaining_tokens_full_for_new_client():
    limiter = TokenBucketRateLimiter(capacity=10.0, rate_per_second=1.0)
    assert limiter.remaining_tokens("new") == 10.0


def test_remaining_tokens_decreases_after_request():
    limiter = TokenBucketRateLimiter(capacity=5.0, rate_per_second=0.01)
    limiter.is_allowed("x")
    assert limiter.remaining_tokens("x") < 5.0


def test_reset_restores_full_capacity():
    limiter = TokenBucketRateLimiter(capacity=3.0, rate_per_second=0.01)
    limiter.is_allowed("r")
    limiter.is_allowed("r")
    limiter.reset("r")
    assert limiter.remaining_tokens("r") == 3.0


def test_clear_removes_all_clients():
    limiter = TokenBucketRateLimiter(capacity=5.0, rate_per_second=1.0)
    limiter.is_allowed("a")
    limiter.is_allowed("b")
    limiter.clear()
    assert limiter.client_count == 0


def test_client_count_tracks_unique_keys():
    limiter = TokenBucketRateLimiter(capacity=5.0, rate_per_second=1.0)
    limiter.is_allowed("alpha")
    limiter.is_allowed("beta")
    limiter.is_allowed("alpha")
    assert limiter.client_count == 2


def test_invalid_capacity_raises():
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(capacity=0.0, rate_per_second=1.0)


def test_invalid_rate_raises():
    with pytest.raises(ValueError):
        TokenBucketRateLimiter(capacity=5.0, rate_per_second=-1.0)


def test_make_rate_limiter_factory():
    limiter = make_rate_limiter(capacity=30.0, rate_per_second=0.5)
    assert isinstance(limiter, TokenBucketRateLimiter)
    assert limiter.is_allowed("x") is True


@pytest.mark.parametrize("capacity", [1.0, 5.0, 10.0, 100.0])
def test_capacity_respected(capacity):
    limiter = TokenBucketRateLimiter(capacity=capacity, rate_per_second=0.001)
    allowed = sum(1 for _ in range(int(capacity) + 5) if limiter.is_allowed("k"))
    assert allowed == int(capacity)


def test_separate_clients_independent():
    limiter = TokenBucketRateLimiter(capacity=1.0, rate_per_second=0.01)
    limiter.is_allowed("client-a")
    assert limiter.is_allowed("client-b") is True

@pytest.mark.parametrize("n_clients", [1, 5, 10])
def test_clear_resets_all_clients(n_clients):
    limiter = TokenBucketRateLimiter(capacity=10.0, rate_per_second=1.0)
    for i in range(n_clients):
        limiter.is_allowed(f"c-{i}")
    limiter.clear()
    assert limiter.client_count == 0


def test_is_allowed_returns_bool():
    limiter = TokenBucketRateLimiter(capacity=5.0, rate_per_second=1.0)
    result = limiter.is_allowed("test")
    assert isinstance(result, bool)


def test_remaining_tokens_after_clear():
    limiter = TokenBucketRateLimiter(capacity=5.0, rate_per_second=0.01)
    limiter.is_allowed("c")
    limiter.clear()
    assert limiter.remaining_tokens("c") == 5.0


@pytest.mark.parametrize("rate", [0.1, 0.5, 1.0, 5.0])
def test_make_rate_limiter_various_rates(rate):
    limiter = make_rate_limiter(capacity=10.0, rate_per_second=rate)
    assert limiter.is_allowed("test") is True
