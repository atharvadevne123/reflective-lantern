"""Schema-based configuration validator.

Validates raw configuration dictionaries against a declarative schema
before they are used by the application, collecting all violations before
raising so callers receive a complete error report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ConfigSchema",
    "FieldSpec",
    "ValidationError",
    "has_required_fields",
    "schema_field_names",
    "validate",
]


@dataclass
class FieldSpec:
    """Specification for a single configuration field."""

    name: str
    type: type
    required: bool = True
    default: Any = None
    choices: list[Any] | None = None
    min_value: float | None = None
    max_value: float | None = None


@dataclass
class ConfigSchema:
    """Collection of field specs that define a valid configuration."""

    fields: list[FieldSpec] = field(default_factory=list)

    def add(self, spec: FieldSpec) -> ConfigSchema:
        self.fields.append(spec)
        return self


class ValidationError(Exception):
    """Raised when configuration validation fails."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("Config validation failed:\n" + "\n".join(f"  - {v}" for v in violations))


def validate(config: dict[str, Any], schema: ConfigSchema) -> dict[str, Any]:
    """Validate *config* against *schema* and return a normalised copy.

    Args:
        config: Raw configuration mapping.
        schema: Schema describing expected fields.

    Returns:
        A new dict with defaults applied for optional missing fields.

    Raises:
        ValidationError: If any field violations are found.
    """
    result: dict[str, Any] = {}
    violations: list[str] = []

    for spec in schema.fields:
        if spec.name not in config:
            if spec.required:
                violations.append(f"'{spec.name}' is required but missing")
            else:
                result[spec.name] = spec.default
            continue

        val = config[spec.name]

        if not isinstance(val, spec.type):
            violations.append(f"'{spec.name}' expected {spec.type.__name__}, got {type(val).__name__}")
            continue

        if spec.choices is not None and val not in spec.choices:
            violations.append(f"'{spec.name}' must be one of {spec.choices}, got {val!r}")

        if spec.min_value is not None and val < spec.min_value:
            violations.append(f"'{spec.name}' must be >= {spec.min_value}, got {val}")

        if spec.max_value is not None and val > spec.max_value:
            violations.append(f"'{spec.name}' must be <= {spec.max_value}, got {val}")

        result[spec.name] = val

    if violations:
        raise ValidationError(violations)

    return result


def schema_field_names(schema: ConfigSchema) -> list[str]:
    """Return a sorted list of all field names defined in *schema*.

    Args:
        schema: A :class:`ConfigSchema` to inspect.

    Returns:
        Sorted list of field name strings.
    """
    return sorted(spec.name for spec in schema.fields)


def has_required_fields(schema: ConfigSchema) -> bool:
    """Return ``True`` if *schema* contains at least one required field.

    Args:
        schema: A :class:`ConfigSchema` to inspect.

    Returns:
        ``True`` when any field has ``required=True``, ``False`` otherwise.
    """
    return any(spec.required for spec in schema.fields)
