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
    "count_fitted_steps",
    "count_pipeline_steps",
    "describe_pipeline",
    "extract_step_classes",
    "first_step",
    "get_step",
    "get_step_names",
    "has_step",
    "is_fitted",
    "last_step",
    "pipeline_has_preprocessor",
    "pipeline_input_features",
    "pipeline_memory_usage_kb",
    "pipeline_param_count",
    "pipeline_step_types",
    "step_is_fitted",
    "step_names_to_set",
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


def count_pipeline_steps(pipeline: Any) -> int:
    """Return the number of steps in a sklearn-style pipeline.

    Args:
        pipeline: A pipeline with a ``steps`` attribute.

    Returns:
        Number of steps, or 0 if no steps attribute.
    """
    steps = getattr(pipeline, "steps", None)
    if steps is None:
        return 0
    return len(steps)


def last_step(pipeline: Any) -> Any:
    """Return the last estimator in a sklearn-style pipeline.

    Args:
        pipeline: A pipeline with a ``steps`` attribute.

    Returns:
        The last step's estimator, or None if the pipeline has no steps.
    """
    steps = getattr(pipeline, "steps", None)
    if not steps:
        return None
    return steps[-1][1]


def step_names_to_set(pipeline: Any) -> set[str]:
    """Return step names as a set for O(1) membership tests.

    Args:
        pipeline: A pipeline with a ``steps`` attribute.

    Returns:
        Set of step name strings.
    """
    steps = getattr(pipeline, "steps", None)
    if steps is None:
        return set()
    return {name for name, _ in steps}


def pipeline_memory_usage_kb(pipeline: Any) -> float:
    """Estimate pickle-based memory footprint of a pipeline in kibibytes.

    Args:
        pipeline: Any picklable object (e.g. a sklearn pipeline).

    Returns:
        Estimated size in KiB.
    """
    import pickle

    data = pickle.dumps(pipeline)
    return round(len(data) / 1024, 4)


def count_fitted_steps(pipeline: Any) -> int:
    """Return the number of steps that have been fitted (have a ``n_features_in_`` attribute).

    Args:
        pipeline: An sklearn Pipeline (fitted or unfitted).

    Returns:
        Count of steps where the underlying estimator has been fitted.
    """
    try:
        return sum(1 for _, estimator in pipeline.steps if hasattr(estimator, "n_features_in_"))
    except AttributeError:
        return 0


def pipeline_has_preprocessor(pipeline: Any) -> bool:
    """Return True if any pipeline step name suggests a preprocessing role.

    Checks step names for common substrings: 'scaler', 'normalizer', 'encoder',
    'imputer', 'transformer'.

    Args:
        pipeline: An sklearn Pipeline.

    Returns:
        True when at least one step name matches a preprocessor keyword.
    """
    keywords = {"scaler", "normalizer", "encoder", "imputer", "transformer"}
    return any(any(kw in name.lower() for kw in keywords) for name in get_step_names(pipeline))


def extract_step_classes(pipeline: Any) -> list[str]:
    """Return a list of estimator class names for each step in *pipeline*.

    Args:
        pipeline: An sklearn Pipeline.

    Returns:
        List of class name strings (e.g. ['StandardScaler', 'LinearRegression']),
        or empty list if *pipeline* has no steps attribute.
    """
    try:
        return [type(estimator).__name__ for _, estimator in pipeline.steps]
    except AttributeError:
        return []


def pipeline_input_features(pipeline: Any) -> list[str] | None:
    """Return the feature names seen during fit, if available.

    Tries ``feature_names_in_`` (sklearn ≥ 1.0) on the pipeline itself, then
    on the first step. Returns None when no feature-name information is stored.

    Args:
        pipeline: A fitted sklearn Pipeline or compatible estimator.

    Returns:
        List of feature name strings, or None if not available.
    """
    for obj in [pipeline, first_step(pipeline)]:
        if obj is None:
            continue
        names = getattr(obj, "feature_names_in_", None)
        if names is not None:
            return list(names)
    return None


def pipeline_step_index(pipeline: Any, step_name: str) -> int:
    """Return the zero-based index of *step_name* in *pipeline*.

    Args:
        pipeline: An sklearn Pipeline.
        step_name: Name of the step to locate.

    Returns:
        Integer index of the step.

    Raises:
        KeyError: If *step_name* is not in the pipeline.
    """
    names = get_step_names(pipeline)
    try:
        return names.index(step_name)
    except ValueError:
        raise KeyError(f"Step {step_name!r} not found in pipeline") from None


def pipeline_step_before(pipeline: Any, step_name: str) -> list[str]:
    """Return the names of all steps that precede *step_name*.

    Args:
        pipeline: An sklearn Pipeline.
        step_name: Reference step name.

    Returns:
        Ordered list of step names before *step_name*; empty if it is the first step.

    Raises:
        KeyError: If *step_name* is not in the pipeline.
    """
    idx = pipeline_step_index(pipeline, step_name)
    return get_step_names(pipeline)[:idx]


def pipeline_step_after(pipeline: Any, step_name: str) -> list[str]:
    """Return the names of all steps that follow *step_name*.

    Args:
        pipeline: An sklearn Pipeline.
        step_name: Reference step name.

    Returns:
        Ordered list of step names after *step_name*; empty if it is the last step.

    Raises:
        KeyError: If *step_name* is not in the pipeline.
    """
    idx = pipeline_step_index(pipeline, step_name)
    return get_step_names(pipeline)[idx + 1 :]


def count_pipeline_params(pipeline: Any) -> int:
    """Count the total number of named parameters across all pipeline steps.

    Args:
        pipeline: A fitted or unfitted sklearn Pipeline.

    Returns:
        Total number of hyperparameter names (from ``get_params(deep=True)``).
    """
    try:
        params = pipeline.get_params(deep=True)
        return len(params)
    except AttributeError:
        return 0


def pipeline_feature_count(pipeline: Any) -> int:
    """Return the number of output features from the pipeline, or 0 if unknown.

    Works with pipelines whose final step exposes ``n_features_in_`` or whose
    penultimate step exposes ``get_feature_names_out``.

    Args:
        pipeline: A fitted sklearn Pipeline.

    Returns:
        Integer feature count, or 0 if it cannot be determined.
    """
    try:
        return int(pipeline.n_features_in_)
    except AttributeError:
        pass  # fall through to alternative lookup
    try:
        return len(pipeline[:-1].get_feature_names_out())
    except Exception:
        return 0


def is_pipeline_fitted(pipeline: Any) -> bool:
    """Check whether a sklearn Pipeline has been fitted.

    Returns True if at least one step exposes a fitted attribute (e.g.
    ``classes_``, ``n_features_in_``, ``feature_importances_``).

    Args:
        pipeline: Any sklearn estimator or Pipeline.

    Returns:
        True if fitted, False otherwise.
    """
    fitted_attrs = ("classes_", "n_features_in_", "feature_importances_", "coef_", "numeric_cols_")
    try:
        steps = list(pipeline.named_steps.values()) if hasattr(pipeline, "named_steps") else [pipeline]
    except Exception:
        steps = [pipeline]
    return any(hasattr(step, attr) for step in steps for attr in fitted_attrs)


def pipeline_step_at(pipeline: Any, index: int) -> Any:
    """Return the step object at *index* in the pipeline's step list.

    Args:
        pipeline: A scikit-learn Pipeline object.
        index: Zero-based index of the desired step.

    Returns:
        The estimator at the given position.

    Raises:
        IndexError: If *index* is out of range.
        AttributeError: If *pipeline* has no ``steps`` attribute.
    """
    steps = pipeline.steps
    if index < 0 or index >= len(steps):
        raise IndexError(f"step index {index} out of range (pipeline has {len(steps)} steps)")
    return steps[index][1]


def pipeline_step_name_at(pipeline: Any, index: int) -> str:
    """Return the step name at *index* in the pipeline.

    Args:
        pipeline: A scikit-learn Pipeline object.
        index: Zero-based index of the desired step.

    Returns:
        Step name as a string.

    Raises:
        IndexError: If *index* is out of range.
        AttributeError: If *pipeline* has no ``steps`` attribute.
    """
    steps = pipeline.steps
    if index < 0 or index >= len(steps):
        raise IndexError(f"step index {index} out of range (pipeline has {len(steps)} steps)")
    return steps[index][0]


def pipeline_estimator_classes(pipeline: Any) -> list[type]:
    """Return a list of the actual class objects for each step.

    Args:
        pipeline: A scikit-learn Pipeline object.

    Returns:
        List of class objects in step order.

    Raises:
        AttributeError: If *pipeline* has no ``steps`` attribute.
    """
    return [type(step) for _, step in pipeline.steps]
