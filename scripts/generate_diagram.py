"""Generate the Logistics-Flow system architecture diagram."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_NAME = "Logistics-Flow"

BOX_STYLE = {"boxstyle": "round,pad=0.5", "linewidth": 1.6}

LAYERS = [
    # (x, y, w, h, label, facecolor, edgecolor)
    (0.4, 5.4, 2.6, 1.0, "Client / Dispatch\nSystem", "#e3f2fd", "#1565c0"),
    (3.6, 5.4, 3.0, 1.0, "FastAPI\n/api/v1/predict\n/health /metrics /drift", "#e8f5e9", "#2e7d32"),
    (7.2, 5.4, 2.6, 1.0, "Pydantic\nValidation +\nCorrelation ID", "#fff8e1", "#f9a825"),
    (10.4, 5.4, 3.0, 1.0, "Rate Limiting\n& OpenAPI Docs", "#f3e5f5", "#6a1b9a"),
    (0.4, 3.4, 3.0, 1.2, "Feature Pipeline\ncyclical time, distance\nbuckets, carrier risk", "#e0f7fa", "#00838f"),
    (4.0, 3.4, 3.2, 1.2, "Ensemble Model\nXGBoost + LightGBM\n+ RandomForest", "#fce4ec", "#c2185b"),
    (7.8, 3.4, 2.6, 1.2, "5-Fold CV\nRMSE / R²", "#ede7f6", "#4527a0"),
    (11.0, 3.4, 2.4, 1.2, "Ensemble\nConfidence", "#e8eaf6", "#283593"),
    (0.4, 1.4, 3.2, 1.2, "Monitoring\nKS-test drift\nprediction logging", "#fff3e0", "#ef6c00"),
    (4.2, 1.4, 3.0, 1.2, "SQLAlchemy\nSQLite dev /\nPostgreSQL prod", "#e0f2f1", "#00695c"),
    (7.8, 1.4, 2.8, 1.2, "Airflow DAG\nweekly retrain", "#f1f8e9", "#558b2f"),
    (11.2, 1.4, 2.2, 1.2, "Docker\nCompose", "#eceff1", "#37474f"),
]

ARROWS = [
    ((3.0, 5.9), (3.6, 5.9)),
    ((6.6, 5.9), (7.2, 5.9)),
    ((9.8, 5.9), (10.4, 5.9)),
    ((5.1, 5.4), (2.0, 4.6)),
    ((2.0, 3.4), (5.6, 2.6)),
    ((5.6, 3.4), (5.7, 2.6)),
    ((7.2, 4.0), (7.8, 4.0)),
    ((10.4, 4.0), (11.0, 4.0)),
    ((3.6, 2.0), (4.2, 2.0)),
    ((7.2, 2.0), (7.8, 2.0)),
    ((9.2, 2.6), (5.6, 3.4)),
]


def main() -> None:
    os.makedirs("screenshots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    for x, y, w, h, label, fc, ec in LAYERS:
        box = mpatches.FancyBboxPatch(
            (x, y), w, h, facecolor=fc, edgecolor=ec, **BOX_STYLE
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="medium",
            color="#212121",
        )

    for (x1, y1), (x2, y2) in ARROWS:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "->", "color": "#546e7a", "lw": 1.3},
        )

    ax.text(
        7, 0.55,
        "API Layer  →  Feature Engineering  →  Ensemble Inference  →  Persistence & Drift Monitoring",
        ha="center", va="center", fontsize=10, style="italic", color="#455a64",
    )

    plt.title(
        f"{PROJECT_NAME} - System Architecture",
        fontsize=15,
        fontweight="bold",
        pad=16,
    )
    plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight")
    print("Wrote screenshots/architecture.png")


if __name__ == "__main__":
    main()
