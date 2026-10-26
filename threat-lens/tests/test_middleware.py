"""Tests for the rate limiter."""

import pytest

from app.middleware import WINDOW_SECONDS, RateLimiter


def test_allows_up_to_limit() -> None:
    limiter = RateLimiter(limit_per_minute=5)
    assert all(limiter.allow("1.2.3.4", now=100.0) for _ in range(5))


def test_blocks_past_limit() -> None:
    limiter = RateLimiter(limit_per_minute=3)
    for _ in range(3):
        assert limiter.allow("1.2.3.4", now=100.0)
    assert limiter.allow("1.2.3.4", now=100.0) is False


def test_clients_are_tracked_independently() -> None:
    limiter = RateLimiter(limit_per_minute=2)
    assert limiter.allow("a", now=10.0)
    assert limiter.allow("a", now=10.0)
    assert limiter.allow("a", now=10.0) is False
    # A different client still has its full allowance.
    assert limiter.allow("b", now=10.0)


def test_window_expiry_frees_capacity() -> None:
    limiter = RateLimiter(limit_per_minute=2)
    assert limiter.allow("a", now=0.0)
    assert limiter.allow("a", now=0.0)
    assert limiter.allow("a", now=0.0) is False
    # Once the window has rolled past, the old hits are discarded.
    assert limiter.allow("a", now=WINDOW_SECONDS + 1.0) is True


def test_remaining_counts_down() -> None:
    limiter = RateLimiter(limit_per_minute=3)
    assert limiter.remaining("a") == 3
    limiter.allow("a", now=1.0)
    assert limiter.remaining("a") == 2


def test_remaining_never_negative() -> None:
    limiter = RateLimiter(limit_per_minute=1)
    limiter.allow("a", now=1.0)
    limiter.allow("a", now=1.0)
    assert limiter.remaining("a") >= 0


def test_reset_clears_state() -> None:
    limiter = RateLimiter(limit_per_minute=1)
    limiter.allow("a", now=1.0)
    limiter.reset()
    assert limiter.allow("a", now=1.0) is True


@pytest.mark.parametrize("limit", [1, 10, 100])
def test_limit_is_respected_for_various_sizes(limit: int) -> None:
    limiter = RateLimiter(limit_per_minute=limit)
    allowed = sum(limiter.allow("x", now=5.0) for _ in range(limit + 5))
    assert allowed == limit
