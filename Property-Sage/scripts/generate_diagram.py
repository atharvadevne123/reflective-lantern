"""Generate architecture diagram for Property-Sage."""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

os.makedirs("screenshots", exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

COLORS = {
    "client": "#3b82f6",
    "api": "#10b981",
    "model": "#8b5cf6",
    "db": "#f59e0b",
    "monitor": "#ef4444",
    "pipeline": "#06b6d4",
    "box_bg": "#1e2130",
    "text": "white",
    "arrow": "#94a3b8",
}


def box(ax, x, y, w, h, label, sublabel="", color="#3b82f6", fontsize=9):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.9,
    )
    ax.add_patch(rect)
    ax.text(x, y + (0.15 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white")
    if sublabel:
        ax.text(x, y - 0.25, sublabel,
                ha="center", va="center", fontsize=6.5, color="#cbd5e1")


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.5))


ax.text(8, 8.6, "Property-Sage — System Architecture", ha="center", va="center",
        fontsize=16, fontweight="bold", color="white")

# Client
box(ax, 1.5, 7, 2.2, 0.9, "Client / Browser", "REST API calls", COLORS["client"])

# FastAPI
box(ax, 5.0, 7, 2.6, 0.9, "FastAPI App", "/predict /health /metrics", COLORS["api"])

# Middleware
box(ax, 5.0, 5.6, 2.2, 0.7, "Middleware", "Correlation ID · Rate limit", "#475569")

# ML Models
box(ax, 9.0, 7.5, 2.8, 0.85, "XGBoost + LightGBM", "Ensemble Price Model", COLORS["model"])
box(ax, 9.0, 6.3, 2.8, 0.85, "RF + XGBoost + LGBM", "Ensemble Rental Model", COLORS["model"])

# Feature Pipeline
box(ax, 5.0, 4.2, 2.6, 0.8, "sklearn Pipeline", "11 engineered features", COLORS["pipeline"])

# SQLite / PostgreSQL
box(ax, 2.0, 4.2, 2.4, 0.8, "PostgreSQL / SQLite", "Predictions · Drift logs", COLORS["db"])

# Monitoring
box(ax, 9.0, 4.8, 2.6, 0.9, "Drift Monitor", "KS-test · 24h window", COLORS["monitor"])

# Retraining
box(ax, 12.5, 5.8, 2.6, 0.85, "Retrain Pipeline", "5-fold CV · Auto-trigger", COLORS["pipeline"])

# Synthetic data
box(ax, 12.5, 4.3, 2.4, 0.75, "Data Generator", "2000 synthetic samples", "#64748b")

# Arrows
arrow(ax, 2.6, 7.0, 3.7, 7.0)     # client -> api
arrow(ax, 5.0, 6.55, 5.0, 5.95)   # api -> middleware
arrow(ax, 5.0, 5.25, 5.0, 4.6)    # middleware -> features
arrow(ax, 6.3, 7.0, 7.6, 7.5)     # api -> price model
arrow(ax, 6.3, 7.0, 7.6, 6.3)     # api -> rental model
arrow(ax, 3.8, 4.2, 2.1 + 1.2, 4.2)  # features -> db
arrow(ax, 2.0, 3.8, 2.0, 1.5)     # db down (store)
arrow(ax, 7.6, 4.8, 6.3, 4.5)     # monitor -> features
arrow(ax, 10.3, 5.0, 11.2, 5.8)   # monitor -> retrain
arrow(ax, 12.5, 5.38, 12.5, 5.0)  # retrain -> data gen
arrow(ax, 11.2, 4.3, 10.3, 4.8)   # data gen -> monitor

# Legend
legend_items = [
    mpatches.Patch(color=COLORS["client"], label="Client"),
    mpatches.Patch(color=COLORS["api"], label="FastAPI"),
    mpatches.Patch(color=COLORS["model"], label="ML Ensemble"),
    mpatches.Patch(color=COLORS["db"], label="Database"),
    mpatches.Patch(color=COLORS["monitor"], label="Monitoring"),
    mpatches.Patch(color=COLORS["pipeline"], label="Pipeline / Retrain"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=8,
          facecolor="#1e2130", edgecolor="#475569", labelcolor="white", ncol=3)

plt.tight_layout()
plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Architecture diagram saved to screenshots/architecture.png")
