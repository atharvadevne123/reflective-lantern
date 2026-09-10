"""Pagination helpers for list endpoints."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from sqlalchemy.orm import Query

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class Page(Generic[T]):
    """A single page of results with pagination metadata."""

    items: list[T]
    total: int
    page: int
    page_size: int

    @property
    def total_pages(self) -> int:
        """Total number of pages given page_size."""
        return max(1, -(-self.total // self.page_size))

    @property
    def has_next(self) -> bool:
        """True if there is at least one more page after this one."""
        return self.page < self.total_pages

    @property
    def has_prev(self) -> bool:
        """True if this is not the first page."""
        return self.page > 1

    def to_dict(self) -> dict[str, Any]:
        """Serialise metadata (not items) to a plain dict."""
        return {
            "total": self.total,
            "page": self.page,
            "page_size": self.page_size,
            "total_pages": self.total_pages,
            "has_next": self.has_next,
            "has_prev": self.has_prev,
        }


def paginate_query(query: Query, page: int = 1, page_size: int = 20) -> Page:
    """Apply LIMIT/OFFSET to a SQLAlchemy query and return a Page.

    Args:
        query: Unexecuted SQLAlchemy query.
        page: 1-indexed page number.
        page_size: Records per page (max 100).

    Returns:
        Page instance with items and metadata.
    """
    page = max(1, page)
    page_size = min(max(1, page_size), 100)
    try:
        total = query.count()
        items = query.offset((page - 1) * page_size).limit(page_size).all()
        logger.debug("Paginated query: total=%d page=%d page_size=%d", total, page, page_size)
        return Page(items=items, total=total, page=page, page_size=page_size)
    except Exception as e:
        logger.error("Pagination failed: %s", e)
        raise
