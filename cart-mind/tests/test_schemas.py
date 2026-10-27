"""Tests for cart-mind app/schemas.py Pydantic models."""

from __future__ import annotations

import pytest

from app.schemas import UserItemFeatures


class TestUserItemFeatures:
    def test_create_minimal(self) -> None:
        f = UserItemFeatures(user_id="u1", item_id="i1")
        assert f.user_id == "u1"
        assert f.item_id == "i1"

    def test_defaults(self) -> None:
        f = UserItemFeatures(user_id="u", item_id="i")
        assert f.purchase_count == 0
        assert f.wishlist_flag == 0

    def test_cart_abandon_rate_range(self) -> None:
        f = UserItemFeatures(user_id="u", item_id="i", cart_abandon_rate=0.5)
        assert f.cart_abandon_rate == pytest.approx(0.5)

    def test_item_avg_rating_range(self) -> None:
        f = UserItemFeatures(user_id="u", item_id="i", item_avg_rating=4.5)
        assert f.item_avg_rating == pytest.approx(4.5)

    def test_invalid_cart_abandon_rate(self) -> None:
        with pytest.raises(Exception):
            UserItemFeatures(user_id="u", item_id="i", cart_abandon_rate=2.0)

    def test_invalid_item_avg_rating(self) -> None:
        with pytest.raises(Exception):
            UserItemFeatures(user_id="u", item_id="i", item_avg_rating=6.0)
