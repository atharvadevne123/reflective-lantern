"""In-process LRU cache helpers for expensive repeated computations."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def lru_cached(maxsize: int = 128) -> Callable[[F], F]:
    """Decorator: wrap a pure function with functools.lru_cache and log cache stats."""

    def decorator(fn: F) -> F:
        cached = functools.lru_cache(maxsize=maxsize)(fn)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = cached(*args, **kwargs)
            info = cached.cache_info()
            logger.debug(
                "cache stats for %s: hits=%d misses=%d maxsize=%d",
                fn.__name__,
                info.hits,
                info.misses,
                info.maxsize,
            )
            return result

        wrapper.cache_info = cached.cache_info  # type: ignore[attr-defined]
        wrapper.cache_clear = cached.cache_clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


@lru_cached(maxsize=4)
def get_regime_label(index: int) -> str:
    """Map regime index to human-readable label (cached)."""
    labels = {0: "bull", 1: "bear", 2: "sideways", 3: "volatile"}
    return labels.get(index, "unknown")


@lru_cached(maxsize=16)
def get_risk_tier(risk_score: float) -> str:
    """Map a 0-1 risk score to a risk tier label (cached)."""
    if risk_score < 0.25:
        return "low"
    if risk_score < 0.5:
        return "moderate"
    if risk_score < 0.75:
        return "high"
    return "critical"
