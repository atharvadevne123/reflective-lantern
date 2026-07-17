"""Utilities for building and inspecting sklearn-compatible pipelines."""

from __future__ import annotations

from typing import Any


def get_step_names(pipeline: Any) -> list[str]:
    """Return the list of step names from a sklearn Pipeline.

    Args:
        pipeline: A fitted or unfitted sklearn Pipeline (or dict bundle).

    Returns:
        List of step name strings, or empty list if not a Pipeline.
    """
    try:
        return [name for name, _ in pipeline.steps]
    except AttributeError:
        return []


def has_step(pipeline: Any, step_name: str) -> bool:
    """Return True if *pipeline* contains a step named *step_name*."""
    return step_name in get_step_names(pipeline)


def get_step(pipeline: Any, step_name: str) -> Any:
    """Return the estimator for *step_name*, or None if not found."""
    try:
        return pipeline.named_steps.get(step_name)
    except AttributeError:
        return None


def pipeline_param_count(pipeline: Any) -> int:
    """Return the total number of parameters (get_params) in the pipeline."""
    try:
        return len(pipeline.get_params())
    except AttributeError:
        return 0


def describe_pipeline(pipeline: Any) -> dict[str, Any]:
    """Return a human-readable description of a Pipeline's steps.

    Includes step names, estimator class names, and parameter counts.
    """
    try:
        steps_info = []
        for name, estimator in pipeline.steps:
            steps_info.append({
                "name": name,
                "class": type(estimator).__name__,
                "n_params": len(estimator.get_params()) if hasattr(estimator, "get_params") else 0,
            })
        return {
            "type": type(pipeline).__name__,
            "n_steps": len(pipeline.steps),
            "steps": steps_info,
        }
    except AttributeError:
        return {"type": type(pipeline).__name__, "n_steps": 0, "steps": []}


def clone_params(pipeline: Any) -> dict[str, Any]:
    """Return a flat dict of all pipeline parameters via get_params(deep=True)."""
    try:
        return dict(pipeline.get_params(deep=True))
    except AttributeError:
        return {}


def bundle_pipeline_info(bundle: dict[str, Any]) -> dict[str, Any]:
    """Extract pipeline description from a model bundle dict.

    Expects bundle to have a ``"model"`` key that is a Pipeline.
    """
    model = bundle.get("model")
    if model is None:
        return {"error": "no model in bundle"}
    return describe_pipeline(model)
