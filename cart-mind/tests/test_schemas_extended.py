"""Extended tests for cart-mind app/schemas.py Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import RecommendRequest, SimilarItemsRequest, UserItemFeatures


VALID_USER_FEATURES = {
    "user_id": "u123",
    "item_id": "i456",
}


class TestUserItemFeatures:
    def test_minimal_valid(self) -> None:
        f = UserItemFeatures(**VALID_USER_FEATURES)
        assert f.user_id == "u123"

    def test_age_above_120_raises(self) -> None:
        with pytest.raises(ValidationError):
            UserItemFeatures(**{**VALID_USER_FEATURES, "user_age": 121})

    def test_age_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            UserItemFeatures(**{**VALID_USER_FEATURES, "user_age": -1})

    def test_cart_abandon_rate_above_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            UserItemFeatures(**{**VALID_USER_FEATURES, "cart_abandon_rate": 1.1})

    def test_item_avg_rating_above_5_raises(self) -> None:
        with pytest.raises(ValidationError):
            UserItemFeatures(**{**VALID_USER_FEATURES, "item_avg_rating": 5.1})

    def test_item_discount_pct_above_100_raises(self) -> None:
        with pytest.raises(ValidationError):
            UserItemFeatures(**{**VALID_USER_FEATURES, "item_discount_pct": 101.0})

    def test_defaults_applied(self) -> None:
        f = UserItemFeatures(**VALID_USER_FEATURES)
        assert f.purchase_count == 0
        assert f.session_count_7d == 0

    def test_negative_purchase_count_raises(self) -> None:
        with pytest.raises(ValidationError):
            UserItemFeatures(**{**VALID_USER_FEATURES, "purchase_count": -1})


class TestRecommendRequest:
    def test_valid_request(self) -> None:
        req = RecommendRequest(user_id="u42", top_k=5)
        assert req.top_k == 5

    def test_top_k_above_50_raises(self) -> None:
        with pytest.raises(ValidationError):
            RecommendRequest(user_id="u42", top_k=51)

    def test_top_k_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            RecommendRequest(user_id="u42", top_k=0)

    def test_default_top_k(self) -> None:
        req = RecommendRequest(user_id="u42")
        assert req.top_k == 10


class TestSimilarItemsRequest:
    def test_valid_request(self) -> None:
        req = SimilarItemsRequest(item_id="i99", top_k=3)
        assert req.item_id == "i99"

    def test_top_k_above_50_raises(self) -> None:
        with pytest.raises(ValidationError):
            SimilarItemsRequest(item_id="i99", top_k=51)

    def test_negative_price_raises(self) -> None:
        with pytest.raises(ValidationError):
            SimilarItemsRequest(item_id="i99", item_price=-1.0)

    def test_default_top_k(self) -> None:
        req = SimilarItemsRequest(item_id="i99")
        assert req.top_k == 5
