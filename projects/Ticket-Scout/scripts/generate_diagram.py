"""Generate the Ticket-Scout system architecture diagram."""

from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

os.makedirs("screenshots", exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#0f1117")

COLORS = {
    "client": "#4c9be8",
    "api": "#2ecc71",
    "models": "#e67e22",
    "db": "#9b59b6",
    "pipeline": "#e74c3c",
    "monitor": "#1abc9c",
    "text": "#ffffff",
    "subtext": "#aaaaaa",
    "arrow": "#555555",
    "border": "#2a2a3a",
}


def box(ax, x, y, w, h, label, sublabel="", color="#2a2a3a", text_color="#ffffff"):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor=text_color,
        linewidth=1.5,
        alpha=0.92,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2 + (0.1 if sublabel else 0),
        label, ha="center", va="center",
        fontsize=9, fontweight="bold", color=text_color,
    )
    if sublabel:
        ax.text(
            x + w / 2, y + h / 2 - 0.22,
            sublabel, ha="center", va="center",
            fontsize=6.5, color=COLORS["subtext"],
        )


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "color": COLORS["arrow"], "lw": 1.5},
    )


# Title
ax.text(8, 8.6, "Ticket-Scout — System Architecture", ha="center", va="center",
        fontsize=14, fontweight="bold", color=COLORS["text"])

# Client layer
box(ax, 0.3, 6.5, 2.2, 1.0, "REST Client", "curl / SDK / UI", color="#1a2a3a", text_color=COLORS["client"])

# FastAPI layer
box(ax, 3.5, 7.0, 2.4, 0.9, "FastAPI", "/predict /health /metrics", color="#1a3a2a", text_color=COLORS["api"])
box(ax, 3.5, 5.8, 2.4, 0.9, "Middleware", "CorrelationID · CORS", color="#1a3a2a", text_color=COLORS["api"])

# Feature engineering
box(ax, 6.8, 7.0, 2.4, 0.9, "Feature Eng.", "TF-IDF + Structured", color="#3a2a1a", text_color=COLORS["models"])

# Models
box(ax, 10.0, 7.6, 2.4, 0.75, "Category Model", "LightGBM 5-class", color="#3a2a1a", text_color=COLORS["models"])
box(ax, 10.0, 6.6, 2.4, 0.75, "Breach Model", "LightGBM binary", color="#3a2a1a", text_color=COLORS["models"])
box(ax, 10.0, 5.6, 2.4, 0.75, "Resolution Model", "LightGBM regressor", color="#3a2a1a", text_color=COLORS["models"])

# Database
box(ax, 3.5, 4.0, 2.4, 0.9, "SQLAlchemy ORM", "SQLite / PostgreSQL", color="#2a1a3a", text_color=COLORS["db"])
box(ax, 6.8, 4.0, 2.4, 0.9, "Prediction Logs", "PredictionLog · DriftLog", color="#2a1a3a", text_color=COLORS["db"])

# Monitoring
box(ax, 6.8, 2.0, 2.4, 0.9, "Drift Monitor", "KS-test · p<0.05", color="#1a3a3a", text_color=COLORS["monitor"])
box(ax, 10.0, 2.0, 2.4, 0.9, "Reference Dist.", "Training baseline", color="#1a3a3a", text_color=COLORS["monitor"])

# Retrain pipeline
box(ax, 3.5, 2.0, 2.4, 0.9, "Retrain Pipeline", "Airflow DAG / CLI", color="#3a1a1a", text_color=COLORS["pipeline"])

# Docker
box(ax, 13.0, 4.5, 2.5, 1.2, "Docker", "API + PostgreSQL\ndocker-compose", color="#1a1a2a", text_color="#aaaaff")

# Arrows
arrow(ax, 2.5, 7.0, 3.5, 7.3)
arrow(ax, 3.5, 6.3, 2.5, 6.7)
arrow(ax, 5.9, 7.3, 6.8, 7.3)
arrow(ax, 9.2, 7.3, 10.0, 7.95)
arrow(ax, 9.2, 7.3, 10.0, 6.95)
arrow(ax, 9.2, 7.3, 10.0, 5.95)
arrow(ax, 5.9, 6.1, 6.8, 4.5)
arrow(ax, 6.8, 4.5, 5.9, 4.5)
arrow(ax, 6.8, 4.0, 6.8, 2.9)
arrow(ax, 6.8, 2.0, 3.5, 2.5)
arrow(ax, 3.5, 2.5, 3.5, 4.0)
arrow(ax, 9.2, 2.0, 10.0, 2.5)

# Legend
ax.text(0.5, 1.2, "Legend:", fontsize=8, color=COLORS["subtext"], fontweight="bold")
for i, (label, color) in enumerate([
    ("FastAPI Layer", COLORS["api"]),
    ("ML Models", COLORS["models"]),
    ("Database", COLORS["db"]),
    ("Monitoring", COLORS["monitor"]),
    ("Pipeline", COLORS["pipeline"]),
]):
    rect = mpatches.Rectangle((0.5 + i * 3.0, 0.7), 0.3, 0.25, facecolor=color, alpha=0.8)
    ax.add_patch(rect)
    ax.text(0.9 + i * 3.0, 0.83, label, fontsize=7, color=COLORS["subtext"])

plt.tight_layout()
plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Architecture diagram saved to screenshots/architecture.png")
