"""Tests for app.pagination module."""

from __future__ import annotations

import pytest

from app.pagination import (
    PageInfo,
    cursor_paginate,
    decode_cursor,
    encode_cursor,
    paginate,
)

ITEMS = list(range(100))
DICT_ITEMS = [{"id": i, "val": i * 2} for i in range(50)]


class TestPageInfo:
    @pytest.mark.parametrize(
        "total,page,per_page,expected_pages",
        [
            (100, 1, 10, 10),
            (0, 1, 10, 0),
            (1, 1, 10, 1),
            (101, 1, 10, 11),
        ],
    )
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

    @pytest.mark.parametrize(
        "data",
        [
            {"id": 0},
            {"id": 999, "extra": True},
            {"key": "string-value"},
        ],
    )
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


class TestPageInfoEdgeCases:
    def test_total_pages_exact_multiple(self):
        info = PageInfo(total=20, page=1, per_page=5)
        assert info.total_pages == 4

    def test_page_beyond_total_pages(self):
        info = PageInfo(total=10, page=99, per_page=5)
        assert info.has_next is False

    @pytest.mark.parametrize("per_page", [1, 5, 10, 25, 100])
    def test_per_page_values(self, per_page: int) -> None:
        info = PageInfo(total=100, page=1, per_page=per_page)
        assert info.total_pages == 100 // per_page


class TestPaginateSliceAccuracy:
    def test_page2_slice_starts_at_correct_offset(self):
        items = list(range(20))
        result = paginate(items, page=2, per_page=5)
        assert result.items == [5, 6, 7, 8, 9]

    def test_all_pages_cover_all_items(self):
        items = list(range(23))
        collected = []
        for p in range(1, 5):
            result = paginate(items, page=p, per_page=7)
            collected.extend(result.items)
        assert collected == items

    def test_single_item_list(self):
        result = paginate([42], page=1, per_page=10)
        assert result.items == [42]
        assert result.info.has_next is False

    def test_page_info_total_always_matches_input_length(self):
        for n in [0, 1, 10, 99]:
            result = paginate(list(range(n)), page=1, per_page=5)
            assert result.info.total == n


class TestCursorPaginateEdgeCases:
    def test_per_page_larger_than_total(self):
        result = cursor_paginate(DICT_ITEMS[:3], cursor=None, per_page=100)
        assert len(result.items) == 3
        assert result.has_next is False

    def test_full_traversal_via_cursors(self):
        n = 22
        per_page = 5
        items = [{"id": i} for i in range(n)]
        collected = []
        cursor = None
        while True:
            result = cursor_paginate(items, cursor=cursor, per_page=per_page)
            collected.extend(result.items)
            if not result.has_next:
                break
            cursor = result.next_cursor
        assert [c["id"] for c in collected] == list(range(n))


class TestPageInfoPageBoundaries:
    @pytest.mark.parametrize(
        "total,page,per_page,has_next,has_prev",
        [
            (10, 1, 10, False, False),
            (10, 2, 5, False, True),
            (10, 1, 5, True, False),
            (0, 1, 10, False, False),
        ],
    )
    def test_has_next_prev_flags(self, total, page, per_page, has_next, has_prev):
        info = PageInfo(total=total, page=page, per_page=per_page)
        assert info.has_next is has_next
        assert info.has_prev is has_prev

    def test_empty_list_has_zero_pages(self):
        info = PageInfo(total=0, page=1, per_page=10)
        assert info.total_pages == 0

    @pytest.mark.parametrize("total,per_page,expected", [(10, 3, 4), (9, 3, 3), (12, 4, 3)])
    def test_total_pages_rounding(self, total, per_page, expected):
        info = PageInfo(total=total, page=1, per_page=per_page)
        assert info.total_pages == expected


class TestEncodeCursorEdgeCases:
    def test_empty_dict_roundtrips(self):
        assert decode_cursor(encode_cursor({})) == {}

    def test_nested_dict_roundtrips(self):
        data = {"nested": {"a": 1}}
        assert decode_cursor(encode_cursor(data)) == data

    @pytest.mark.parametrize("key,val", [("id", 0), ("name", ""), ("flag", True)])
    def test_various_value_types(self, key, val):
        data = {key: val}
        assert decode_cursor(encode_cursor(data)) == data


class TestPaginateInfoField:
    def test_info_is_page_info_instance(self):
        page = paginate(ITEMS, page=1, per_page=10)
        assert isinstance(page.info, PageInfo)

    def test_info_page_matches_requested_page(self):
        page = paginate(ITEMS, page=3, per_page=10)
        assert page.info.page == 3

    def test_info_per_page_matches_requested(self):
        page = paginate(ITEMS, page=1, per_page=15)
        assert page.info.per_page == 15
