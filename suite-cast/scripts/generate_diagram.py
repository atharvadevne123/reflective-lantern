"""Generate the Suite-Cast system architecture diagram as a PNG."""

from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def draw_box(ax, x, y, w, h, label, color, fontsize=10):
    """Draw a rounded box with centered label."""
    box = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.08",
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="#1a1a1a",
    )


def draw_arrow(ax, x1, y1, x2, y2, label=""):
    """Draw an arrow between two points with an optional midpoint label."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.6},
    )
    if label:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.12,
            label,
            ha="center",
            fontsize=8,
            color="#444444",
            style="italic",
        )


def main() -> None:
    """Render architecture.png into the screenshots directory."""
    os.makedirs("screenshots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    plt.title("Suite-Cast — System Architecture", fontsize=15, fontweight="bold", pad=16)

    # Client layer
    draw_box(ax, 0.3, 5.0, 2.4, 1.2, "Hotel PMS /\nBooking Client", "#cfe8ff")

    # API layer
    draw_box(ax, 3.6, 5.0, 2.6, 1.2, "FastAPI\n/api/v1/predict", "#d8f2d8")
    draw_box(ax, 3.6, 3.3, 2.6, 1.0, "Rate Limit +\nCorrelation ID", "#eef3d8")
    draw_box(ax, 3.6, 1.7, 2.6, 1.0, "/health  /metrics", "#d8f2d8")

    # Feature + model layer
    draw_box(ax, 7.2, 5.0, 2.8, 1.2, "Feature Pipeline\n(18 features)", "#ffe9c7")
    draw_box(ax, 7.2, 3.3, 2.8, 1.2, "XGBoost + LightGBM\nEnsemble (5-fold CV)", "#ffd9d9")
    draw_box(ax, 7.2, 1.5, 2.8, 1.2, "Drift Monitor\n(KS-test)", "#f3d8f2")

    # Storage + ops layer
    draw_box(ax, 11.0, 5.0, 2.6, 1.2, "PostgreSQL\nprediction log", "#d9e2f3")
    draw_box(ax, 11.0, 3.3, 2.6, 1.2, "Retraining DAG\n(weekly, Airflow)", "#e2e2e2")
    draw_box(ax, 11.0, 1.5, 2.6, 1.2, "Model Artifacts\n(joblib + metrics)", "#fdf3c7")

    # Arrows
    draw_arrow(ax, 2.7, 5.6, 3.6, 5.6, "POST")
    draw_arrow(ax, 6.2, 5.6, 7.2, 5.6, "features")
    draw_arrow(ax, 8.6, 5.0, 8.6, 4.5, "X")
    draw_arrow(ax, 10.0, 5.6, 11.0, 5.6, "log")
    draw_arrow(ax, 8.6, 3.3, 8.6, 2.7, "scores")
    draw_arrow(ax, 10.0, 3.9, 11.0, 3.9, "trigger")
    draw_arrow(ax, 11.0, 2.7, 10.0, 2.2, "")
    draw_arrow(ax, 4.9, 5.0, 4.9, 4.3, "")
    draw_arrow(ax, 4.9, 3.3, 4.9, 2.7, "")

    plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight")
    print("Wrote screenshots/architecture.png")


if __name__ == "__main__":
    main()
