"""Tests for fleet-mind app/parts_inventory.py."""

from __future__ import annotations

import pytest

from app.parts_inventory import Inventory, Part, reorder_suggestions


def _part(pid: str, qty: int = 10, reorder: int = 5, unit_cost: float = 100.0) -> Part:
    return Part(part_id=pid, name=f"Part {pid}", unit_cost=unit_cost, quantity_on_hand=qty, reorder_point=reorder, reorder_quantity=20)


class TestInventoryConsume:
    def test_consume_reduces_stock(self) -> None:
        inv = Inventory()
        inv.add_part(_part("P1", qty=10))
        result = inv.consume("P1", 3)
        assert result is True
        assert inv.get_part("P1").quantity_on_hand == 7

    def test_consume_insufficient_stock(self) -> None:
        inv = Inventory()
        inv.add_part(_part("P1", qty=2))
        result = inv.consume("P1", 5)
        assert result is False

    def test_consume_missing_part(self) -> None:
        inv = Inventory()
        assert inv.consume("MISSING", 1) is False


class TestInventoryRestock:
    def test_restock_adds_quantity(self) -> None:
        inv = Inventory()
        inv.add_part(_part("P1", qty=5))
        inv.restock("P1", 10)
        assert inv.get_part("P1").quantity_on_hand == 15


class TestBelowReorderPoint:
    def test_returns_low_stock_parts(self) -> None:
        inv = Inventory()
        inv.add_part(_part("P1", qty=3, reorder=5))
        inv.add_part(_part("P2", qty=10, reorder=5))
        low = inv.below_reorder_point()
        assert len(low) == 1
        assert low[0].part_id == "P1"


class TestTotalValue:
    def test_calculates_correctly(self) -> None:
        inv = Inventory()
        inv.add_part(_part("P1", qty=5, unit_cost=100.0))
        inv.add_part(_part("P2", qty=2, unit_cost=50.0))
        assert inv.total_value() == pytest.approx(600.0)

    def test_empty_inventory(self) -> None:
        assert Inventory().total_value() == pytest.approx(0.0)


class TestReorderSuggestions:
    def test_suggests_low_stock_parts(self) -> None:
        inv = Inventory()
        inv.add_part(_part("P1", qty=2, reorder=5))
        suggestions = reorder_suggestions(inv)
        assert len(suggestions) == 1
        assert suggestions[0]["part_id"] == "P1"

    def test_no_suggestions_when_well_stocked(self) -> None:
        inv = Inventory()
        inv.add_part(_part("P1", qty=20, reorder=5))
        assert reorder_suggestions(inv) == []
