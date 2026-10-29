"""Pagination utilities for list endpoints."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class PaginationParams(BaseModel):
    """Query parameters for paginated list endpoints."""

    page: int = Field(default=1, ge=1, description="1-indexed page number")
    page_size: int = Field(default=20, ge=1, le=200, description="Results per page")

    @property
    def offset(self) -> int:
        """Return the SQL OFFSET value for this page."""
        return (self.page - 1) * self.page_size


class Page(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""

    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int

    @classmethod
    def from_items(cls, items: list[T], total: int, params: PaginationParams) -> "Page[T]":
        """Construct a Page from a slice of items and pagination params.

        Args:
            items: The page's items (already sliced).
            total: Total number of items across all pages.
            params: The pagination parameters that produced this page.

        Returns:
            A populated Page instance.
        """
        total_pages = max(1, -(-total // params.page_size))  # ceiling division
        return cls(
            items=items,
            total=total,
            page=params.page,
            page_size=params.page_size,
            total_pages=total_pages,
        )
