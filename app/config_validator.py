"""Schema-based configuration validator.

Validates raw configuration dictionaries against a declarative schema
before they are used by the application, collecting all violations before
raising so callers receive a complete error report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

__all__ = [
    "FieldSpec",
    "ConfigSchema",
    "ValidationError",
    "validate",
]


@dataclass
class FieldSpec:
    """Specification for a single configuration field."""

    name: str
    type: Type
    required: bool = True
    default: Any = None
    choices: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class ConfigSchema:
    """Collection of field specs that define a valid configuration."""

    fields: List[FieldSpec] = field(default_factory=list)

    def add(self, spec: FieldSpec) -> ConfigSchema:
        self.fields.append(spec)
        return self


class ValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, violations: List[str]) -> None:
        self.violations = violations
        super().__init__("Config validation failed:\n" + "\n".join(f"  - {v}" for v in violations))


def validate(config: Dict[str, Any], schema: ConfigSchema) -> Dict[str, Any]:
    """Validate *config* against *schema* and return a normalised copy.

    Args:
        config: Raw configuration mapping.
        schema: Schema describing expected fields.

    Returns:
        A new dict with defaults applied for optional missing fields.

    Raises:
        ValidationError: If any field violations are found.
    """
    result: Dict[str, Any] = {}
    violations: List[str] = []

    for spec in schema.fields:
        if spec.name not in config:
            if spec.required:
                violations.append(f"'{spec.name}' is required but missing")
            else:
                result[spec.name] = spec.default
            continue

        val = config[spec.name]

        if not isinstance(val, spec.type):
            violations.append(
                f"'{spec.name}' expected {spec.type.__name__}, got {type(val).__name__}"
            )
            continue

        if spec.choices is not None and val not in spec.choices:
            violations.append(
                f"'{spec.name}' must be one of {spec.choices}, got {val!r}"
            )

        if spec.min_value is not None and val < spec.min_value:
            violations.append(f"'{spec.name}' must be >= {spec.min_value}, got {val}")

        if spec.max_value is not None and val > spec.max_value:
            violations.append(f"'{spec.name}' must be <= {spec.max_value}, got {val}")

        result[spec.name] = val

    if violations:
        raise ValidationError(violations)

    return result
