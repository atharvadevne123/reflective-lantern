"""
Reporting utilities for Price-Prophet evaluation results.

Functions here turn raw metric dicts into human-readable summaries,
compare multiple models, and persist results to disk as JSON.
"""

from __future__ import annotations

import json
import os


def format_metrics_report(metrics: dict, title: str = "Metrics Report") -> str:
    """Format a metrics dict as a human-readable string with an optional title.

    Parameters
    ----------
    metrics:
        Dict of metric name → value.
    title:
        Report heading (default ``"Metrics Report"``).

    Returns
    -------
    str
        Multi-line formatted report.
    """
    lines = [title, "=" * max(len(title), 40)]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def build_backtest_report(result: dict) -> str:
    """Build a formatted string report from a backtest result dict.

    Parameters
    ----------
    result:
        Dict expected to contain keys such as ``baseline_revenue``,
        ``optimised_revenue``, ``revenue_uplift_pct``, ``mae``, ``n_periods``.

    Returns
    -------
    str
        Multi-line report summarising the backtest outcome.
    """
    uplift = result.get("revenue_uplift_pct", 0.0)
    n = result.get("n_periods", 0)
    baseline = result.get("baseline_revenue", 0.0)
    optimised = result.get("optimised_revenue", 0.0)
    mae = result.get("mae", 0.0)
    lines = [
        "Backtest Report",
        "=" * 40,
        f"  Periods evaluated : {n}",
        f"  Baseline revenue  : {baseline:.2f}",
        f"  Optimised revenue : {optimised:.2f}",
        f"  Revenue uplift    : {uplift:.2f}%",
        f"  MAE               : {mae:.4f}",
    ]
    return "\n".join(lines)


def generate_summary(metrics: dict) -> str:
    """Format a metrics dict into a human-readable multi-line string.

    Unknown keys are included in the output so that custom metrics are
    not silently swallowed.

    Parameters
    ----------
    metrics:
        Dict of metric name → value, e.g. as returned by the
        :mod:`app.evaluation.metrics` functions or the backtester.

    Returns
    -------
    str
        Multi-line report suitable for printing to a terminal or log.
    """
    lines = ["=" * 40, "Evaluation Summary", "=" * 40]

    # Well-known keys get pretty labels; everything else uses the raw key.
    _LABELS: dict[str, str] = {
        "mae": "MAE (Mean Absolute Error)",
        "rmse": "RMSE",
        "mape": "MAPE (%)",
        "r_squared": "R² Score",
        "revenue_uplift": "Revenue Uplift (%)",
        "n_windows": "Windows evaluated",
        "n_samples": "Samples",
    }

    for key, value in metrics.items():
        label = _LABELS.get(key, key.replace("_", " ").title())
        if isinstance(value, float):
            lines.append(f"  {label:<30s}: {value:.4f}")
        else:
            lines.append(f"  {label:<30s}: {value}")

    lines.append("=" * 40)
    return "\n".join(lines)


def compare_models(results: dict[str, dict]) -> dict[str, int]:
    """Rank models by MAE (lower is better).

    Parameters
    ----------
    results:
        Mapping of model name → metrics dict.  Each dict must contain
        a ``"mae"`` key.

    Returns
    -------
    dict
        Mapping of model name → 1-indexed rank, where rank 1 is the
        best (lowest MAE).
    """
    if not results:
        return {}

    # Sort by MAE ascending; models without "mae" go to the end.
    sorted_names = sorted(
        results.keys(),
        key=lambda name: results[name].get("mae", float("inf")),
    )

    return {name: rank + 1 for rank, name in enumerate(sorted_names)}


def compare_models_by_metric(
    results: dict[str, dict], metric: str = "mae"
) -> list[tuple[str, float]]:
    """Rank models by a given metric and return sorted (name, value) pairs.

    Parameters
    ----------
    results:
        Mapping of model name → metrics dict.
    metric:
        Key to rank by (default ``"mae"``).  Lower is better for error
        metrics; models lacking the key are placed last.

    Returns
    -------
    list[tuple[str, float]]
        Pairs ``(model_name, metric_value)`` sorted ascending by value.
    """
    ranked = sorted(
        results.items(),
        key=lambda item: item[1].get(metric, float("inf")),
    )
    return [(name, metrics.get(metric, float("nan"))) for name, metrics in ranked]


def metrics_diff(metrics_a: dict, metrics_b: dict) -> dict[str, float]:
    """Compute the numeric difference (A - B) for shared numeric keys.

    Parameters
    ----------
    metrics_a:
        Baseline metrics dict.
    metrics_b:
        Comparison metrics dict.

    Returns
    -------
    dict
        Mapping of shared key → ``value_a - value_b`` for all keys
        present in both dicts with numeric values.
    """
    result: dict[str, float] = {}
    for key in metrics_a:
        if key in metrics_b:
            a, b = metrics_a[key], metrics_b[key]
            if isinstance(a, int | float) and isinstance(b, int | float):
                result[key] = round(float(a) - float(b), 6)
    return result


def export_metrics(metrics: dict, path: str) -> None:
    """Write *metrics* to *path* as a formatted JSON file.

    Parent directories are created automatically.

    Parameters
    ----------
    metrics:
        Metrics dict to serialise.
    path:
        Destination file path (e.g. ``"reports/run_001.json"``).
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=str)
