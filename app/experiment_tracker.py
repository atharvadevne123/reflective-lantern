"""Simple A/B experiment tracker for ML model comparisons."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Variant:
    """A single experiment variant.

    Attributes:
        name: Variant identifier (e.g. 'control', 'treatment').
        weight: Relative traffic weight (default 1.0).
        metadata: Arbitrary key-value config for this variant.
    """

    name: str
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Outcome of a single experiment evaluation.

    Attributes:
        experiment: Experiment name.
        variant: Assigned variant name.
        entity_id: The entity that was assigned.
    """

    experiment: str
    variant: str
    entity_id: str


class Experiment:
    """Manages variant assignment for one experiment.

    Assignment is deterministic: the same entity_id always receives
    the same variant within a given experiment, based on a weighted hash.

    Args:
        name: Unique experiment name.
        variants: List of :class:`Variant` objects.
        enabled: Whether the experiment is active.
    """

    def __init__(
        self,
        name: str,
        variants: list[Variant],
        enabled: bool = True,
    ) -> None:
        if not variants:
            raise ValueError("At least one variant is required")
        self.name = name
        self.variants = variants
        self.enabled = enabled
        self._total_weight = sum(v.weight for v in variants)
        self._assignment_counts: dict[str, int] = {v.name: 0 for v in variants}

    def assign(self, entity_id: str) -> ExperimentResult:
        """Assign an entity to a variant deterministically.

        Args:
            entity_id: Unique identifier for the entity (user ID, session, etc.).

        Returns:
            :class:`ExperimentResult` with the assigned variant.
        """
        if not self.enabled:
            variant = self.variants[0]
            logger.debug("Experiment '%s' disabled; assigning to '%s'", self.name, variant.name)
            return ExperimentResult(self.name, variant.name, entity_id)

        key = f"{self.name}:{entity_id}"
        digest = int(hashlib.md5(key.encode()).hexdigest(), 16)
        bucket = (digest % 10_000) / 10_000.0 * self._total_weight
        cumulative = 0.0
        selected = self.variants[-1]
        for variant in self.variants:
            cumulative += variant.weight
            if bucket < cumulative:
                selected = variant
                break
        self._assignment_counts[selected.name] += 1
        logger.debug(
            "Experiment '%s': entity '%s' -> variant '%s'",
            self.name,
            entity_id,
            selected.name,
        )
        return ExperimentResult(self.name, selected.name, entity_id)

    def assignment_distribution(self) -> dict[str, int]:
        """Return how many entities were assigned to each variant."""
        return dict(self._assignment_counts)


class ExperimentRegistry:
    """Registry of named experiments."""

    def __init__(self) -> None:
        self._experiments: dict[str, Experiment] = {}

    def register(self, experiment: Experiment) -> None:
        """Add an experiment to the registry."""
        self._experiments[experiment.name] = experiment
        logger.info("Registered experiment '%s'", experiment.name)

    def get(self, name: str) -> Experiment | None:
        """Retrieve an experiment by name."""
        return self._experiments.get(name)

    def assign(self, experiment_name: str, entity_id: str) -> ExperimentResult | None:
        """Assign an entity to a variant in the named experiment.

        Returns None if the experiment is not found.
        """
        exp = self._experiments.get(experiment_name)
        if exp is None:
            logger.warning("Experiment '%s' not found", experiment_name)
            return None
        return exp.assign(entity_id)

    def list_experiments(self) -> list[str]:
        """Return names of all registered experiments."""
        return list(self._experiments)


__all__ = [
    "Experiment",
    "ExperimentRegistry",
    "ExperimentResult",
    "Variant",
]
