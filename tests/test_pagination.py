"""Tests for app.pagination module."""

from __future__ import annotations

import pytest

from app.pagination import (
    CursorPage,
    Page,
    PageInfo,
    cursor_paginate,
    decode_cursor,
    encode_cursor,
    paginate,
)

ITEMS = list(range(100))
DICT_ITEMS = [{"id": i, "val": i * 2} for i in range(50)]


class TestPageInfo:
    @pytest.mark.parametrize("total,page,per_page,expected_pages", [
        (100, 1, 10, 10),
        (0, 1, 10, 0),
        (1, 1, 10, 1),
        (101, 1, 10, 11),
    ])
    def test_total_pages(self, total, page, per_page, expected_pages):
        info = PageInfo(total=total, page=page, per_page=per_page)
        assert info.total_pages == expected_pages

    def test_has_next_and_prev(self):
        info = PageInfo(total=30, page=2, per_page=10)
        assert info.has_next is True
        assert info.has_prev is True

    def test_first_page_no_prev(self):
        info = PageInfo(total=30, page=1, per_page=10)
        assert info.has_prev is False

    def test_last_page_no_next(self):
        info = PageInfo(total=30, page=3, per_page=10)
        assert info.has_next is False


class TestPaginate:
    def test_first_page_returns_correct_slice(self):
        page = paginate(ITEMS, page=1, per_page=10)
        assert page.items == list(range(10))
        assert page.info.total == 100

    def test_last_page_partial(self):
        page = paginate(list(range(25)), page=3, per_page=10)
        assert page.items == [20, 21, 22, 23, 24]

    def test_empty_list(self):
        page = paginate([], page=1, per_page=10)
        assert page.items == []
        assert page.info.total == 0

    def test_raises_on_invalid_page(self):
        with pytest.raises(ValueError, match="page must be"):
            paginate(ITEMS, page=0)

    def test_raises_on_invalid_per_page(self):
        with pytest.raises(ValueError, match="per_page must be"):
            paginate(ITEMS, per_page=0)

    @pytest.mark.parametrize("page,per_page", [(1, 5), (2, 5), (3, 5)])
    def test_various_pages(self, page, per_page):
        result = paginate(list(range(15)), page=page, per_page=per_page)
        assert len(result.items) == per_page


class TestCursorEncoding:
    def test_roundtrip(self):
        data = {"id": 42, "name": "test"}
        assert decode_cursor(encode_cursor(data)) == data

    def test_invalid_cursor_raises(self):
        with pytest.raises(ValueError, match="Invalid cursor"):
            decode_cursor("!!!not-base64!!!")

    @pytest.mark.parametrize("data", [
        {"id": 0},
        {"id": 999, "extra": True},
        {"key": "string-value"},
    ])
    def test_various_payloads(self, data):
        assert decode_cursor(encode_cursor(data)) == data


class TestCursorPaginate:
    def test_first_page_no_cursor(self):
        result = cursor_paginate(DICT_ITEMS, cursor=None, per_page=10)
        assert len(result.items) == 10
        assert result.items[0]["id"] == 0
        assert result.has_next is True

    def test_subsequent_page_via_cursor(self):
        first = cursor_paginate(DICT_ITEMS, cursor=None, per_page=5)
        second = cursor_paginate(DICT_ITEMS, cursor=first.next_cursor, per_page=5)
        assert second.items[0]["id"] == 5

    def test_last_page_no_next_cursor(self):
        result = cursor_paginate(DICT_ITEMS[:5], cursor=None, per_page=10)
        assert result.has_next is False
        assert result.next_cursor is None

    def test_empty_list(self):
        result = cursor_paginate([], per_page=10)
        assert result.items == []
        assert result.has_next is False
