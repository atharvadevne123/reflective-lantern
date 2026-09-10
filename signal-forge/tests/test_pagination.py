"""Tests for pagination helper."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.pagination import Page


def test_page_total_pages_exact():
    p = Page(items=[], total=40, page=1, page_size=20)
    assert p.total_pages == 2


def test_page_total_pages_remainder():
    p = Page(items=[], total=41, page=1, page_size=20)
    assert p.total_pages == 3


def test_page_has_next_true():
    p = Page(items=[], total=40, page=1, page_size=20)
    assert p.has_next is True


def test_page_has_next_last_page():
    p = Page(items=[], total=40, page=2, page_size=20)
    assert p.has_next is False


def test_page_has_prev_first_page():
    p = Page(items=[], total=100, page=1, page_size=20)
    assert p.has_prev is False


def test_page_has_prev_second_page():
    p = Page(items=[], total=100, page=2, page_size=20)
    assert p.has_prev is True


def test_page_to_dict_keys():
    p = Page(items=[], total=50, page=2, page_size=10)
    d = p.to_dict()
    assert set(d.keys()) == {"total", "page", "page_size", "total_pages", "has_next", "has_prev"}


@pytest.mark.parametrize("total,page_size,expected_pages", [
    (0, 10, 1),
    (10, 10, 1),
    (11, 10, 2),
    (100, 20, 5),
])
def test_page_total_pages_parametrize(total, page_size, expected_pages):
    p = Page(items=[], total=total, page=1, page_size=page_size)
    assert p.total_pages == expected_pages
