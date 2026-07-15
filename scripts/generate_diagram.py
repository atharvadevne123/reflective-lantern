"""Generate the Watt-Guard system architecture diagram."""

from __future__ import annotations

import logging
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

logger = logging.getLogger(__name__)
os.makedirs("screenshots", exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#0f1923")

COLORS = {
    "api": "#1e88e5",
    "model": "#43a047",
    "store": "#fb8c00",
    "monitor": "#8e24aa",
    "pipe": "#e53935",
    "ext": "#546e7a",
    "text": "white",
}


def box(ax: plt.Axes, x: float, y: float, w: float, h: float, label: str, color: str, fontsize: int = 9) -> None:
    """Draw a rounded rectangle with centred label text."""
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        color="white",
        fontsize=fontsize,
        fontweight="bold",
        zorder=4,
    )


def arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    """Draw an annotated arrow between two points."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.2},
        zorder=5,
    )


# Title
ax.text(
    8, 8.5, "Watt-Guard — System Architecture", ha="center", va="center", color="white", fontsize=14, fontweight="bold"
)

# Clients
box(ax, 0.3, 6.5, 1.8, 0.8, "Building\nSensors", COLORS["ext"])
box(ax, 0.3, 5.4, 1.8, 0.8, "SCADA /\nIoT Hub", COLORS["ext"])

# FastAPI
box(ax, 2.8, 5.5, 2.5, 2.0, "FastAPI\n/predict\n/anomaly\n/drift\n/metrics", COLORS["api"], fontsize=8)

# Feature Pipeline
box(ax, 6.2, 6.5, 2.2, 1.0, "Feature\nPipeline", COLORS["model"])

# ML Models
box(ax, 6.2, 5.0, 2.2, 1.2, "XGBoost +\nLightGBM +\nRandForest", COLORS["model"], fontsize=8)

# Anomaly
box(ax, 9.2, 6.5, 2.2, 1.0, "Isolation\nForest", COLORS["monitor"])

# KS Drift
box(ax, 9.2, 5.0, 2.2, 1.2, "KS-Drift\nDetector", COLORS["monitor"])

# DB
box(ax, 6.8, 2.8, 2.0, 1.2, "SQLite /\nPostgreSQL\n(SQLAlchemy)", COLORS["store"], fontsize=8)

# Airflow
box(ax, 9.5, 2.8, 2.2, 1.2, "Airflow DAG\nWeekly\nRetrain", COLORS["pipe"], fontsize=8)

# Docker
box(ax, 12.0, 5.5, 2.2, 1.5, "Docker\nCompose\n(API+DB)", COLORS["ext"])

# CI
box(ax, 12.0, 3.5, 2.2, 1.2, "GitHub\nActions CI\nruff+pytest", COLORS["ext"], fontsize=8)

# Arrows
arrow(ax, 2.1, 7.1, 2.8, 6.8)
arrow(ax, 2.1, 5.8, 2.8, 6.0)
arrow(ax, 5.3, 6.7, 6.2, 7.0)
arrow(ax, 5.3, 6.3, 6.2, 5.8)
arrow(ax, 5.3, 6.3, 9.2, 7.0)
arrow(ax, 5.3, 6.3, 9.2, 5.6)
arrow(ax, 7.8, 5.0, 7.8, 4.0)
arrow(ax, 10.3, 5.0, 10.3, 4.0)
arrow(ax, 9.5, 3.4, 8.8, 3.5)

# Legend
legend_patches = [
    mpatches.Patch(color=COLORS["api"], label="REST API"),
    mpatches.Patch(color=COLORS["model"], label="ML Models"),
    mpatches.Patch(color=COLORS["monitor"], label="Monitoring"),
    mpatches.Patch(color=COLORS["store"], label="Storage"),
    mpatches.Patch(color=COLORS["pipe"], label="Pipelines"),
    mpatches.Patch(color=COLORS["ext"], label="Infrastructure"),
]
ax.legend(handles=legend_patches, loc="lower left", fontsize=8, framealpha=0.3, labelcolor="white", facecolor="#1a2a3a")

plt.tight_layout()
plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight")
logger.info("Architecture diagram saved to screenshots/architecture.png")
