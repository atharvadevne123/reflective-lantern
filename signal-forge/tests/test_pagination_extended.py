"""Extended tests for signal-forge app/pagination.py."""

from __future__ import annotations

from app.pagination import Page


class TestPageProperties:
    def _page(self, total: int, page: int = 1, page_size: int = 10) -> Page:
        items = list(range(min(page_size, total)))
        return Page(items=items, total=total, page=page, page_size=page_size)

    def test_total_pages_single(self) -> None:
        p = self._page(total=5, page_size=10)
        assert p.total_pages == 1

    def test_total_pages_multiple(self) -> None:
        p = self._page(total=25, page_size=10)
        assert p.total_pages == 3

    def test_total_pages_exact(self) -> None:
        p = self._page(total=20, page_size=10)
        assert p.total_pages == 2

    def test_total_pages_zero_total(self) -> None:
        p = self._page(total=0, page_size=10)
        assert p.total_pages == 1

    def test_has_next_true(self) -> None:
        p = self._page(total=25, page=1, page_size=10)
        assert p.has_next is True

    def test_has_next_false_last_page(self) -> None:
        p = self._page(total=25, page=3, page_size=10)
        assert p.has_next is False

    def test_has_prev_false_first_page(self) -> None:
        p = self._page(total=25, page=1, page_size=10)
        assert p.has_prev is False

    def test_has_prev_true_later_page(self) -> None:
        p = self._page(total=25, page=2, page_size=10)
        assert p.has_prev is True

    def test_to_dict_keys(self) -> None:
        p = self._page(total=25, page=1, page_size=10)
        d = p.to_dict()
        assert "total" in d
        assert "page" in d
        assert "page_size" in d
        assert "total_pages" in d
        assert "has_next" in d
        assert "has_prev" in d

    def test_to_dict_values(self) -> None:
        p = self._page(total=25, page=2, page_size=10)
        d = p.to_dict()
        assert d["total"] == 25
        assert d["page"] == 2
        assert d["page_size"] == 10

    def test_items_in_page(self) -> None:
        p = Page(items=["a", "b", "c"], total=3, page=1, page_size=3)
        assert len(p.items) == 3

    def test_ceil_division_page_count(self) -> None:
        p = self._page(total=11, page_size=10)
        assert p.total_pages == 2
