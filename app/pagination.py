"""Pagination utilities for API endpoints."""

from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class PageInfo:
    """Metadata about a paginated response page.

    Attributes:
        total: Total number of items across all pages.
        page: Current 1-based page number.
        per_page: Items per page.
        total_pages: Total number of pages.
        has_next: Whether a next page exists.
        has_prev: Whether a previous page exists.
    """

    total: int
    page: int
    per_page: int
    total_pages: int = field(init=False)
    has_next: bool = field(init=False)
    has_prev: bool = field(init=False)

    def __post_init__(self) -> None:
        self.total_pages = math.ceil(self.total / self.per_page) if self.per_page > 0 else 0
        self.has_next = self.page < self.total_pages
        self.has_prev = self.page > 1


@dataclass
class Page(Generic[T]):
    """A single page of results with metadata.

    Attributes:
        items: List of items on this page.
        info: Pagination metadata.
    """

    items: list[T]
    info: PageInfo


def paginate(items: list, page: int = 1, per_page: int = 20) -> Page:
    """Slice a list into a page.

    Args:
        items: Full list of items to paginate.
        page: 1-based page number.
        per_page: Number of items per page.

    Returns:
        A Page with the requested slice and metadata.

    Raises:
        ValueError: If page < 1 or per_page < 1.
    """
    if page < 1:
        raise ValueError(f"page must be >= 1, got {page}")
    if per_page < 1:
        raise ValueError(f"per_page must be >= 1, got {per_page}")
    total = len(items)
    start = (page - 1) * per_page
    end = start + per_page
    return Page(items=items[start:end], info=PageInfo(total=total, page=page, per_page=per_page))


def encode_cursor(data: dict[str, Any]) -> str:
    """Encode a dict into a URL-safe base64 cursor token.

    Args:
        data: Arbitrary dict to encode (must be JSON-serialisable).

    Returns:
        URL-safe base64 string.
    """
    return base64.urlsafe_b64encode(json.dumps(data, separators=(",", ":")).encode()).decode()


def decode_cursor(cursor: str) -> dict[str, Any]:
    """Decode a cursor token back into a dict.

    Args:
        cursor: URL-safe base64 string produced by :func:`encode_cursor`.

    Returns:
        Original dict.

    Raises:
        ValueError: If cursor is malformed.
    """
    try:
        return json.loads(base64.urlsafe_b64decode(cursor.encode()))
    except Exception as exc:
        raise ValueError(f"Invalid cursor: {exc}") from exc


@dataclass
class CursorPage(Generic[T]):
    """Cursor-based page of results.

    Attributes:
        items: Items on this page.
        next_cursor: Opaque token for the next page; None if no more pages.
        has_next: Whether a next page exists.
    """

    items: list[T]
    next_cursor: str | None
    has_next: bool


def cursor_paginate(
    items: list,
    cursor: str | None = None,
    per_page: int = 20,
    key: str = "id",
) -> CursorPage:
    """Paginate a sorted list using an opaque cursor.

    Args:
        items: Full sorted list of items (each must be a dict with ``key``).
        cursor: Opaque cursor from a previous response; None for first page.
        per_page: Maximum items to return.
        key: Dict key used for cursor positioning.

    Returns:
        A CursorPage with items and next cursor.
    """
    start_idx = 0
    if cursor is not None:
        decoded = decode_cursor(cursor)
        last_val = decoded.get(key)
        for idx, item in enumerate(items):
            if isinstance(item, dict) and item.get(key) == last_val:
                start_idx = idx + 1
                break
    page_items = items[start_idx : start_idx + per_page]
    has_next = (start_idx + per_page) < len(items)
    next_cursor: str | None = None
    if has_next and page_items:
        last_item = page_items[-1]
        if isinstance(last_item, dict):
            next_cursor = encode_cursor({key: last_item.get(key)})
    return CursorPage(items=page_items, next_cursor=next_cursor, has_next=has_next)


def page_range(info: PageInfo, window: int = 5) -> list[int]:
    """Return a list of page numbers centred on *info.page* for navigation.

    Args:
        info: PageInfo from a paginated response.
        window: Total number of page numbers to show (capped by total_pages).

    Returns:
        Sorted list of 1-based page numbers.
    """
    half = window // 2
    start = max(1, info.page - half)
    end = min(info.total_pages, start + window - 1)
    start = max(1, end - window + 1)
    return list(range(start, end + 1))


def last_page_items(items: list, per_page: int) -> int:
    """Return how many items appear on the final page.

    Args:
        items: Full list of items.
        per_page: Page size.

    Returns:
        Remainder count (1–per_page), or 0 if items is empty.
    """
    if not items or per_page < 1:
        return 0
    remainder = len(items) % per_page
    return remainder if remainder else per_page


__all__ = [
    "CursorPage",
    "Page",
    "PageInfo",
    "cursor_paginate",
    "decode_cursor",
    "encode_cursor",
    "last_page_items",
    "page_range",
    "paginate",
]
