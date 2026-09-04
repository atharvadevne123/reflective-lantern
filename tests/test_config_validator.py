"""Tests for app.config_validator."""

import pytest

from app.config_validator import ConfigSchema, FieldSpec, ValidationError, validate

# ---------------------------------------------------------------------------
# Parametrized helpers
# ---------------------------------------------------------------------------


VALID_PORTS = [1, 80, 443, 8080, 8443, 65535]
INVALID_PORTS = [0, -1, 65536, 99999]
VALID_ENVS = ["dev", "prod"]
INVALID_ENVS = ["staging", "qa", "", "DEV", "PROD"]


def make_schema() -> ConfigSchema:
    schema = ConfigSchema()
    schema.add(FieldSpec("host", str, required=True))
    schema.add(FieldSpec("port", int, required=True, min_value=1, max_value=65535))
    schema.add(FieldSpec("debug", bool, required=False, default=False))
    schema.add(FieldSpec("env", str, required=True, choices=["dev", "prod"]))
    return schema


class TestValidate:
    def test_valid_config(self):
        schema = make_schema()
        result = validate({"host": "localhost", "port": 8080, "env": "dev"}, schema)
        assert result["host"] == "localhost"
        assert result["port"] == 8080
        assert result["debug"] is False

    def test_missing_required_raises(self):
        schema = make_schema()
        with pytest.raises(ValidationError) as exc_info:
            validate({"host": "localhost", "port": 8080}, schema)
        assert "'env'" in str(exc_info.value)

    def test_wrong_type_raises(self):
        schema = make_schema()
        with pytest.raises(ValidationError) as exc_info:
            validate({"host": 123, "port": 8080, "env": "dev"}, schema)
        assert "'host'" in str(exc_info.value)

    def test_invalid_choice_raises(self):
        schema = make_schema()
        with pytest.raises(ValidationError) as exc_info:
            validate({"host": "x", "port": 8080, "env": "staging"}, schema)
        assert "'env'" in str(exc_info.value)

    def test_below_min_raises(self):
        schema = make_schema()
        with pytest.raises(ValidationError) as exc_info:
            validate({"host": "x", "port": 0, "env": "dev"}, schema)
        assert "'port'" in str(exc_info.value)

    def test_above_max_raises(self):
        schema = make_schema()
        with pytest.raises(ValidationError) as exc_info:
            validate({"host": "x", "port": 99999, "env": "dev"}, schema)
        assert "'port'" in str(exc_info.value)

    def test_multiple_violations_collected(self):
        schema = make_schema()
        with pytest.raises(ValidationError) as exc_info:
            validate({}, schema)
        # host, port, env are all required
        assert len(exc_info.value.violations) >= 3

    def test_default_applied_for_optional(self):
        schema = make_schema()
        result = validate({"host": "h", "port": 80, "env": "prod"}, schema)
        assert result["debug"] is False

    @pytest.mark.parametrize("port", VALID_PORTS)
    def test_valid_port_boundaries(self, port: int) -> None:
        schema = make_schema()
        result = validate({"host": "x", "port": port, "env": "dev"}, schema)
        assert result["port"] == port

    @pytest.mark.parametrize("port", INVALID_PORTS)
    def test_invalid_port_boundaries(self, port: int) -> None:
        schema = make_schema()
        with pytest.raises(ValidationError) as exc_info:
            validate({"host": "x", "port": port, "env": "dev"}, schema)
        assert "'port'" in str(exc_info.value)

    @pytest.mark.parametrize("env", VALID_ENVS)
    def test_valid_env_choices(self, env: str) -> None:
        schema = make_schema()
        result = validate({"host": "x", "port": 80, "env": env}, schema)
        assert result["env"] == env

    @pytest.mark.parametrize("env", INVALID_ENVS)
    def test_invalid_env_choices(self, env: str) -> None:
        schema = make_schema()
        with pytest.raises(ValidationError) as exc_info:
            validate({"host": "x", "port": 80, "env": env}, schema)
        assert "'env'" in str(exc_info.value)

    def test_extra_keys_are_ignored(self) -> None:
        """Keys not declared in the schema should be silently dropped."""
        schema = make_schema()
        result = validate({"host": "h", "port": 80, "env": "dev", "unknown_key": "oops"}, schema)
        assert "unknown_key" not in result

    def test_schema_add_returns_self_for_chaining(self) -> None:
        schema = ConfigSchema()
        returned = schema.add(FieldSpec("x", str))
        assert returned is schema

    def test_validation_error_message_contains_all_violations(self) -> None:
        schema = make_schema()
        with pytest.raises(ValidationError) as exc_info:
            validate({"port": 8080}, schema)
        msg = str(exc_info.value)
        assert "'host'" in msg
        assert "'env'" in msg

    def test_optional_field_with_explicit_none_default(self) -> None:
        schema = ConfigSchema()
        schema.add(FieldSpec("timeout", float, required=False, default=None))
        result = validate({}, schema)
        assert result["timeout"] is None

    def test_exact_min_value_accepted(self) -> None:
        schema = make_schema()
        result = validate({"host": "h", "port": 1, "env": "prod"}, schema)
        assert result["port"] == 1

    def test_exact_max_value_accepted(self) -> None:
        schema = make_schema()
        result = validate({"host": "h", "port": 65535, "env": "prod"}, schema)
        assert result["port"] == 65535
