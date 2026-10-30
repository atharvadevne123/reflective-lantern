"""Tests for validate_year_built_value."""

import pytest

from app.validators import validate_year_built_value


def test_valid_year():
    assert validate_year_built_value(2000) == 2000


def test_ancient_year_raises():
    with pytest.raises(ValueError, match="1800"):
        validate_year_built_value(1799)


def test_future_year_raises():
    with pytest.raises(ValueError, match="future"):
        validate_year_built_value(2100)


@pytest.mark.parametrize("year", [1850, 1950, 2000, 2010, 2024])
def test_valid_years_parametrized(year):
    assert validate_year_built_value(year) == year
