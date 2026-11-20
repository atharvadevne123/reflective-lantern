"""Tests for app.config_validator."""

import pytest

from app.config_validator import ConfigSchema, FieldSpec, ValidationError, validate


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
