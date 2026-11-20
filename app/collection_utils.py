"""Collection and iterable utility functions for the Watt-Guard application."""

from __future__ import annotations

from collections import Counter
from typing import Any, TypeVar

T = TypeVar("T")

__all__ = [
    "chunk",
    "deep_merge",
    "flatten",
    "frequency_map",
    "unique_ordered",
]


def chunk(items: list[T], size: int) -> list[list[T]]:
    """Split *items* into consecutive chunks of at most *size* elements.

    Args:
        items: The list to split.
        size: Maximum number of elements per chunk. Must be >= 1.

    Returns:
        List of sub-lists. The last chunk may be smaller than *size*.

    Raises:
        ValueError: If *size* < 1.
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    return [items[i : i + size] for i in range(0, len(items), size)]


def flatten(nested: list[Any]) -> list[Any]:
    """Flatten one level of nesting from *nested*.

    Args:
        nested: A list that may contain sub-lists or plain elements.

    Returns:
        A new flat list containing all elements, sub-list items expanded.
    """
    result: list[Any] = []
    for item in nested:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


def unique_ordered(items: list[T]) -> list[T]:
    """Return *items* with duplicates removed while preserving insertion order.

    Args:
        items: Input list potentially containing duplicates.

    Returns:
        New list with first occurrence of each element kept in order.
    """
    seen: set[Any] = set()
    result: list[T] = []
    for item in items:
        try:
            key = item
            if key not in seen:
                seen.add(key)
                result.append(item)
        except TypeError:
            result.append(item)
    return result


def frequency_map(items: list[Any]) -> dict[Any, int]:
    """Return a mapping of each unique element in *items* to its occurrence count.

    Args:
        items: List of hashable elements to count.

    Returns:
        Dict mapping element -> count, sorted descending by count.
    """
    counts = Counter(items)
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into a copy of *base*.

    Keys present only in *base* are kept; keys present only in *override* are
    added; keys in both are resolved recursively for nested dicts or by
    replacing with the *override* value otherwise.

    Args:
        base: The base dictionary.
        override: Values in this dict take precedence over *base*.

    Returns:
        New merged dictionary; neither *base* nor *override* is mutated.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
