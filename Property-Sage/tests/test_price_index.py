"""Tests for NeighborhoodPriceIndex."""

import pytest

from app.price_index import NeighborhoodPriceIndex


@pytest.fixture
def index():
    idx = NeighborhoodPriceIndex()
    idx.add_sample("suburb", 300_000.0)
    idx.add_sample("suburb", 400_000.0)
    idx.add_sample("downtown", 600_000.0)
    return idx


def test_get_median(index):
    assert index.get_median("suburb") == 350_000.0


def test_get_mean(index):
    assert index.get_mean("suburb") == 350_000.0


def test_unknown_neighborhood(index):
    assert index.get_median("moon_base") is None


def test_all_neighborhoods(index):
    nbs = index.all_neighborhoods()
    assert "suburb" in nbs
    assert "downtown" in nbs


def test_index_summary(index):
    summary = index.index_summary()
    assert "suburb" in summary
    assert summary["suburb"]["count"] == 2


@pytest.mark.parametrize("nb,price", [("midtown", 500_000.0), ("waterfront", 800_000.0)])
def test_add_sample(nb, price):
    idx = NeighborhoodPriceIndex()
    idx.add_sample(nb, price)
    assert idx.get_median(nb) == price
