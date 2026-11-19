"""Parts inventory management for fleet maintenance."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Part:
    part_id: str
    name: str
    unit_cost: float
    quantity_on_hand: int
    reorder_point: int
    reorder_quantity: int


@dataclass
class Inventory:
    parts: dict[str, Part] = field(default_factory=dict)

    def add_part(self, part: Part) -> None:
        self.parts[part.part_id] = part

    def get_part(self, part_id: str) -> Part | None:
        return self.parts.get(part_id)

    def consume(self, part_id: str, quantity: int) -> bool:
        """Reduce stock. Returns False if insufficient quantity."""
        part = self.parts.get(part_id)
        if part is None or part.quantity_on_hand < quantity:
            return False
        part.quantity_on_hand -= quantity
        return True

    def restock(self, part_id: str, quantity: int) -> None:
        part = self.parts.get(part_id)
        if part is not None:
            part.quantity_on_hand += quantity

    def below_reorder_point(self) -> list[Part]:
        return [p for p in self.parts.values() if p.quantity_on_hand <= p.reorder_point]

    def total_value(self) -> float:
        return sum(p.unit_cost * p.quantity_on_hand for p in self.parts.values())


def reorder_suggestions(inventory: Inventory) -> list[dict[str, object]]:
    """Return suggested reorder orders for parts at or below reorder point."""
    suggestions = []
    for part in inventory.below_reorder_point():
        suggestions.append({
            "part_id": part.part_id,
            "name": part.name,
            "current_qty": part.quantity_on_hand,
            "reorder_qty": part.reorder_quantity,
            "estimated_cost": part.unit_cost * part.reorder_quantity,
        })
    return suggestions
