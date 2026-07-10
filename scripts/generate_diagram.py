"""Generate the Traffic-Pulse system architecture diagram."""

from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    color: str,
    fontsize: int = 8,
) -> None:
    patch = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.12",
        facecolor=color,
        edgecolor="white",
        alpha=0.92,
        linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="white",
        fontweight="bold",
    )


def main() -> None:
    os.makedirs("screenshots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_facecolor("#12122a")
    fig.patch.set_facecolor("#12122a")
    ax.axis("off")

    ax.text(
        8,
        8.6,
        "Traffic-Pulse — System Architecture",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="white",
    )

    _box(ax, 0.2, 6.6, 1.8, 1.1, "REST Client\n(SDK / curl)", "#455A64")
    _box(
        ax,
        3.2,
        5.7,
        2.8,
        2.8,
        "FastAPI App\n/api/v1/predict\n/api/v1/drift\n/api/v1/metrics\n/health",
        "#1565C0",
    )
    _box(ax, 7.2, 7.2, 2.5, 1.2, "Feature\nEngineering\n(26 features)", "#00695C")
    _box(ax, 7.2, 5.5, 2.5, 1.5, "XGBoost +\nLightGBM\nEnsemble", "#4A148C")
    _box(ax, 7.2, 3.7, 2.5, 1.5, "Monitoring\nKS-Test Drift\nPrediction Log", "#B71C1C")
    _box(ax, 10.8, 6.8, 2.4, 1.2, "SQLite / PostgreSQL\n(SQLAlchemy ORM)", "#4E342E")
    _box(ax, 10.8, 4.8, 2.4, 1.6, "Retraining Pipeline\nretrain_dag.py\n(@weekly)", "#E65100")
    _box(ax, 3.2, 3.0, 2.8, 1.5, "Docker\ndocker-compose\n(API + PostgreSQL)", "#006064")
    _box(ax, 7.2, 1.5, 2.5, 1.5, "GitHub Actions CI\nruff lint + pytest\n(on push / PR)", "#37474F")

    ap = {"arrowstyle": "->", "color": "#90CAF9", "lw": 1.4}
    arrows = [
        (2.0, 7.15, 3.2, 7.0),
        (6.0, 7.0, 7.2, 7.8),
        (6.0, 6.5, 7.2, 6.2),
        (6.0, 5.8, 7.2, 4.5),
        (9.7, 7.8, 10.8, 7.3),
        (9.7, 6.2, 10.8, 7.0),
        (9.7, 4.5, 10.8, 5.5),
        (4.6, 5.7, 4.6, 4.5),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=ap)

    legend = [
        mpatches.Patch(color="#1565C0", label="API Layer"),
        mpatches.Patch(color="#4A148C", label="ML Ensemble"),
        mpatches.Patch(color="#00695C", label="Feature Engineering"),
        mpatches.Patch(color="#B71C1C", label="Monitoring / Drift"),
        mpatches.Patch(color="#4E342E", label="Database"),
        mpatches.Patch(color="#E65100", label="Retraining"),
        mpatches.Patch(color="#006064", label="Infrastructure"),
    ]
    ax.legend(
        handles=legend,
        loc="lower left",
        fontsize=8,
        facecolor="#1e1e3a",
        edgecolor="white",
        labelcolor="white",
    )

    plt.tight_layout()
    plt.savefig(
        "screenshots/architecture.png",
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    print("Saved screenshots/architecture.png")


if __name__ == "__main__":
    main()
