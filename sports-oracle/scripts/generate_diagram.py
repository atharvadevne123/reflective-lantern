"""Generate the Sports-Oracle architecture diagram."""

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
ax.set_facecolor("#0d1117")

COLORS = {
    "title": "#58a6ff",
    "client": "#3fb950",
    "api": "#388bfd",
    "ml": "#f78166",
    "data": "#d2a8ff",
    "infra": "#ffa657",
    "monitor": "#79c0ff",
    "arrow": "#8b949e",
    "box_bg": "#161b22",
    "box_border": "#30363d",
    "text": "#c9d1d9",
    "subtitle": "#8b949e",
}


def box(ax, x, y, w, h, title, subtitle="", color="#388bfd"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        facecolor=COLORS["box_bg"],
        edgecolor=color,
        linewidth=2.0,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h * 0.65, title,
        ha="center", va="center", fontsize=8.5, fontweight="bold",
        color=color, zorder=4,
    )
    if subtitle:
        ax.text(
            x + w / 2, y + h * 0.28, subtitle,
            ha="center", va="center", fontsize=6.5,
            color=COLORS["subtitle"], zorder=4,
        )


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.5),
        zorder=5,
    )


# Title
ax.text(8, 8.55, "Sports-Oracle — System Architecture", ha="center", va="center",
        fontsize=14, fontweight="bold", color=COLORS["title"])

# Client
box(ax, 0.3, 6.8, 2.0, 1.0, "API Client", "REST / cURL / SDK", COLORS["client"])

# FastAPI layer
box(ax, 3.0, 6.0, 2.4, 1.8, "FastAPI", "/predict  /batch\n/metrics  /drift\n/health  /retrain", COLORS["api"])

# ML Pipeline
box(ax, 6.5, 5.8, 3.0, 2.2, "ML Pipeline", "XGBoost + LightGBM\n+ RandomForest\nVotingClassifier", COLORS["ml"])

# Feature Engineering
box(ax, 6.5, 3.3, 3.0, 2.0, "Feature Pipeline", "FormIndex · H2H\nAttack/Defense\nRest Days · Drop", COLORS["ml"])

# Monitoring
box(ax, 3.0, 3.5, 2.4, 1.8, "Monitoring", "KS-test + PSI\nDrift Detection\nPrediction Log", COLORS["monitor"])

# Database
box(ax, 0.3, 2.8, 2.0, 1.6, "PostgreSQL", "matches\nprediction_logs\ndrift_logs", COLORS["data"])

# Airflow
box(ax, 10.5, 5.8, 2.4, 2.0, "Airflow DAG", "Weekly Retrain\nChampion/Challenger\nAUC Gate ≥ 0.65", COLORS["infra"])

# Model Store
box(ax, 10.5, 3.3, 2.4, 2.0, "Model Store", "model.joblib\nmetrics.json\nchallenger.joblib", COLORS["infra"])

# GitHub Actions
box(ax, 13.3, 5.8, 2.4, 2.0, "GitHub Actions", "ruff lint\npytest 3.11/3.12\nwheel build", COLORS["infra"])

# Arrows
arrow(ax, 2.3, 7.3, 3.0, 7.1)
arrow(ax, 5.4, 7.0, 6.5, 7.0)
arrow(ax, 5.4, 6.5, 6.5, 6.5)
arrow(ax, 8.0, 5.8, 8.0, 5.3)
arrow(ax, 5.4, 4.4, 3.0, 4.4)
arrow(ax, 3.0, 4.0, 0.3 + 2.0, 4.0)
arrow(ax, 9.5, 6.8, 10.5, 6.8)
arrow(ax, 10.5 + 1.2, 5.8, 10.5 + 1.2, 5.3)
arrow(ax, 12.9, 6.8, 13.3, 6.8)

# Legend
legend_items = [
    mpatches.Patch(facecolor=COLORS["box_bg"], edgecolor=COLORS["client"], label="Client"),
    mpatches.Patch(facecolor=COLORS["box_bg"], edgecolor=COLORS["api"], label="API Layer"),
    mpatches.Patch(facecolor=COLORS["box_bg"], edgecolor=COLORS["ml"], label="ML / Features"),
    mpatches.Patch(facecolor=COLORS["box_bg"], edgecolor=COLORS["monitor"], label="Monitoring"),
    mpatches.Patch(facecolor=COLORS["box_bg"], edgecolor=COLORS["data"], label="Database"),
    mpatches.Patch(facecolor=COLORS["box_bg"], edgecolor=COLORS["infra"], label="Infrastructure"),
]
ax.legend(
    handles=legend_items, loc="lower right",
    facecolor=COLORS["box_bg"], edgecolor=COLORS["box_border"],
    labelcolor=COLORS["text"], fontsize=7.5,
)

plt.tight_layout()
plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Architecture diagram saved to screenshots/architecture.png")
