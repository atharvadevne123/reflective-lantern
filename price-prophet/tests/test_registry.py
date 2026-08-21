"""Tests for app/models/registry.py."""

from __future__ import annotations

import pytest


def test_registry_register_get():
    from app.models.registry import ModelRegistry

    reg = ModelRegistry()
    reg.register("m1", object())
    assert reg.get("m1") is not None


def test_registry_get_missing():
    from app.models.registry import ModelRegistry

    reg = ModelRegistry()
    with pytest.raises(KeyError):
        reg.get("missing")


def test_registry_list_models():
    from app.models.registry import ModelRegistry

    reg = ModelRegistry()
    reg.register("b", 1)
    reg.register("a", 2)
    assert reg.list_models() == ["a", "b"]


def test_registry_remove():
    from app.models.registry import ModelRegistry

    reg = ModelRegistry()
    reg.register("m", 1)
    removed = reg.remove("m")
    assert removed is True
    assert len(reg) == 0


def test_registry_remove_missing():
    from app.models.registry import ModelRegistry

    reg = ModelRegistry()
    assert reg.remove("nope") is False


def test_registry_len():
    from app.models.registry import ModelRegistry

    reg = ModelRegistry()
    assert len(reg) == 0
    reg.register("x", 1)
    assert len(reg) == 1


def test_registry_contains_operator() -> None:
    from app.models.registry import ModelRegistry

    reg = ModelRegistry()
    reg.register("m1", object())
    assert "m1" in reg
    assert "m2" not in reg


def test_registry_remove_nonexistent_returns_false() -> None:
    from app.models.registry import ModelRegistry

    reg = ModelRegistry()
    assert reg.remove("nonexistent") is False


def test_registry_list_sorted() -> None:
    from app.models.registry import ModelRegistry

    reg = ModelRegistry()
    reg.register("z_model", object())
    reg.register("a_model", object())
    names = reg.list_models()
    assert names == sorted(names)


def test_registry_len_grows_with_registrations() -> None:
    from app.models.registry import ModelRegistry

    reg = ModelRegistry()
    assert len(reg) == 0
    reg.register("m1", object())
    reg.register("m2", object())
    assert len(reg) == 2
