"""Generate Claim-Cast system architecture diagram."""

from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

os.makedirs("screenshots", exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#0d1117")

COLORS = {
    "client": "#1f6feb",
    "api": "#3fb950",
    "model": "#d29922",
    "db": "#8957e5",
    "monitor": "#f78166",
    "pipeline": "#58a6ff",
    "arrow": "#8b949e",
    "text": "#e6edf3",
    "bg_dark": "#161b22",
}


def box(ax, x, y, w, h, label, sublabel, color):
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.08",
        facecolor=color + "33",
        edgecolor=color,
        linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h * 0.65,
        label,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=color,
    )
    ax.text(
        x + w / 2,
        y + h * 0.28,
        sublabel,
        ha="center",
        va="center",
        fontsize=6.5,
        color=COLORS["text"] + "aa",
    )


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.2),
    )


ax.text(
    8,
    8.5,
    "Claim-Cast — System Architecture",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color=COLORS["text"],
)

box(ax, 0.3, 6.5, 2.0, 1.2, "Client", "HTTP / JSON", COLORS["client"])
box(ax, 3.0, 6.5, 2.4, 1.2, "FastAPI", "/predict /health\n/metrics /drift /premium", COLORS["api"])
box(
    ax,
    6.5,
    6.5,
    2.4,
    1.2,
    "XGBoost Pipeline",
    "5-fold CV · 12 features\nActuarial engineering",
    COLORS["model"],
)
box(
    ax,
    10.0,
    6.5,
    2.4,
    1.2,
    "Isolation Forest",
    "Anomaly detection\nOutlier claims",
    COLORS["model"],
)
box(ax, 3.0, 4.5, 2.4, 1.2, "SQLAlchemy ORM", "SQLite (dev)\nPostgreSQL (prod)", COLORS["db"])
box(
    ax,
    6.5,
    4.5,
    2.4,
    1.2,
    "KS-Test Monitor",
    "Drift detection\nPrediction logging",
    COLORS["monitor"],
)
box(
    ax, 10.0, 4.5, 2.4, 1.2, "Retraining DAG", "Airflow weekly\nModel promotion", COLORS["pipeline"]
)
box(ax, 3.0, 2.3, 2.4, 1.2, "PostgreSQL", "claim_predictions\ndrift_logs", COLORS["db"])
box(ax, 6.5, 2.3, 2.4, 1.2, "Structlog", "Correlation IDs\nRate limiting", COLORS["api"])
box(ax, 10.0, 2.3, 2.4, 1.2, "Docker + CI", "docker-compose\nGitHub Actions", COLORS["pipeline"])

arrow(ax, 2.3, 7.1, 3.0, 7.1)
arrow(ax, 5.4, 7.1, 6.5, 7.1)
arrow(ax, 8.9, 7.1, 10.0, 7.1)
arrow(ax, 4.2, 6.5, 4.2, 5.7)
arrow(ax, 7.7, 6.5, 7.7, 5.7)
arrow(ax, 11.2, 6.5, 11.2, 5.7)
arrow(ax, 4.2, 4.5, 4.2, 3.5)
arrow(ax, 7.7, 4.5, 7.7, 3.5)
arrow(ax, 11.2, 4.5, 11.2, 3.5)

legend_items = [
    mpatches.Patch(facecolor=COLORS["client"] + "33", edgecolor=COLORS["client"], label="Client"),
    mpatches.Patch(facecolor=COLORS["api"] + "33", edgecolor=COLORS["api"], label="API Layer"),
    mpatches.Patch(facecolor=COLORS["model"] + "33", edgecolor=COLORS["model"], label="ML Models"),
    mpatches.Patch(facecolor=COLORS["db"] + "33", edgecolor=COLORS["db"], label="Database"),
    mpatches.Patch(
        facecolor=COLORS["monitor"] + "33", edgecolor=COLORS["monitor"], label="Monitoring"
    ),
    mpatches.Patch(
        facecolor=COLORS["pipeline"] + "33", edgecolor=COLORS["pipeline"], label="Infrastructure"
    ),
]
ax.legend(
    handles=legend_items,
    loc="lower left",
    fontsize=7,
    framealpha=0.3,
    labelcolor=COLORS["text"],
    facecolor=COLORS["bg_dark"],
)

plt.tight_layout()
plt.savefig(
    "screenshots/architecture.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
)
print("Architecture diagram saved to screenshots/architecture.png")
