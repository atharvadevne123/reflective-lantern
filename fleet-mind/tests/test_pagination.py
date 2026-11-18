"""Tests for Fleet-Mind pagination utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pagination import Page, PaginationParams


def test_pagination_params_defaults():
    params = PaginationParams()
    assert params.page == 1
    assert params.page_size == 20


def test_pagination_params_offset_first_page():
    params = PaginationParams(page=1, page_size=20)
    assert params.offset == 0


def test_pagination_params_offset_second_page():
    params = PaginationParams(page=2, page_size=20)
    assert params.offset == 20


def test_pagination_params_offset_third_page():
    params = PaginationParams(page=3, page_size=10)
    assert params.offset == 20


def test_pagination_params_min_page_size():
    params = PaginationParams(page=1, page_size=1)
    assert params.page_size == 1


def test_pagination_params_max_page_size():
    params = PaginationParams(page=1, page_size=200)
    assert params.page_size == 200


def test_pagination_params_invalid_page():
    with pytest.raises(Exception):
        PaginationParams(page=0)


def test_pagination_params_invalid_page_size():
    with pytest.raises(Exception):
        PaginationParams(page_size=0)


def test_page_from_items_total_pages():
    params = PaginationParams(page=1, page_size=10)
    page = Page.from_items(items=list(range(10)), total=25, params=params)
    assert page.total_pages == 3


def test_page_from_items_exact_division():
    params = PaginationParams(page=1, page_size=5)
    page = Page.from_items(items=list(range(5)), total=20, params=params)
    assert page.total_pages == 4


def test_page_from_items_total_less_than_page_size():
    params = PaginationParams(page=1, page_size=20)
    page = Page.from_items(items=[1, 2, 3], total=3, params=params)
    assert page.total_pages == 1


def test_page_from_items_zero_total():
    params = PaginationParams(page=1, page_size=10)
    page = Page.from_items(items=[], total=0, params=params)
    assert page.total_pages == 1
    assert page.total == 0


def test_page_from_items_carries_params():
    params = PaginationParams(page=2, page_size=15)
    page = Page.from_items(items=list(range(15)), total=45, params=params)
    assert page.page == 2
    assert page.page_size == 15


@pytest.mark.parametrize(
    "total,page_size,expected",
    [
        (100, 10, 10),
        (101, 10, 11),
        (1, 10, 1),
        (0, 10, 1),
    ],
)
def test_total_pages_calculation(total, page_size, expected):
    params = PaginationParams(page=1, page_size=page_size)
    page = Page.from_items(items=[], total=total, params=params)
    assert page.total_pages == expected
