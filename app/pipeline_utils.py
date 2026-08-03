"""Utilities for building and inspecting sklearn-compatible pipelines."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
            steps_info.append(
                {
                    "name": name,
                    "class": type(estimator).__name__,
                    "n_params": len(estimator.get_params()) if hasattr(estimator, "get_params") else 0,
                }
            )
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

    Expects bundle to have a ``"pipeline"`` or ``"model"`` key that is a Pipeline.
    Falls back gracefully when neither key is present.
    """
    pipeline = bundle.get("pipeline") or bundle.get("model")
    if pipeline is None:
        logger.warning("bundle_pipeline_info: no 'pipeline' or 'model' key in bundle")
        return {"error": "no pipeline in bundle"}
    info = describe_pipeline(pipeline)
    logger.debug("bundle_pipeline_info: steps=%d", len(info.get("steps", [])))
    return info


def is_fitted(estimator: Any) -> bool:
    """Return True if *estimator* appears to have been fitted.

    Checks for the presence of attributes ending in ``_`` (sklearn convention).

    Args:
        estimator: Any sklearn-compatible estimator or pipeline.

    Returns:
        True when at least one fitted attribute is found.
    """
    fitted_attrs = [a for a in dir(estimator) if a.endswith("_") and not a.startswith("__")]
    return len(fitted_attrs) > 0


def step_is_fitted(pipeline: Any, step_name: str) -> bool:
    """Return True if the named step within *pipeline* has been fitted.

    Args:
        pipeline: sklearn Pipeline or compatible object.
        step_name: Name of the pipeline step to check.

    Returns:
        True when the step exists and appears fitted; False otherwise.
    """
    step = get_step(pipeline, step_name)
    if step is None:
        return False
    return is_fitted(step)


__all__ = [
    "bundle_pipeline_info",
    "clone_params",
    "describe_pipeline",
    "first_step",
    "get_step",
    "get_step_names",
    "has_step",
    "is_fitted",
    "pipeline_param_count",
    "pipeline_step_types",
    "step_is_fitted",
]


def pipeline_step_types(pipeline: Any) -> dict[str, str]:
    """Return a mapping of step name to transformer class name.

    Args:
        pipeline: sklearn Pipeline or object with a ``steps`` attribute.

    Returns:
        Dict of step_name → class_name, or empty dict if no steps.
    """
    steps = getattr(pipeline, "steps", None)
    if not steps:
        return {}
    return {name: type(step).__name__ for name, step in steps}


def first_step(pipeline: Any) -> Any:
    """Return the first transformer in *pipeline*'s step list.

    Args:
        pipeline: sklearn Pipeline or object with a ``steps`` attribute.

    Returns:
        The first step object, or None if the pipeline has no steps.
    """
    steps = getattr(pipeline, "steps", None)
    if not steps:
        return None
    return steps[0][1]
