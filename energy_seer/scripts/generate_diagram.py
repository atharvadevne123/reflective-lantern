"""Generate Energy-Seer system architecture diagram."""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.makedirs("screenshots", exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

COLORS = {
    "api": "#1f6feb",
    "model": "#238636",
    "monitoring": "#d29922",
    "storage": "#6e40c9",
    "pipeline": "#f78166",
    "rag": "#56d364",
    "client": "#58a6ff",
    "arrow": "#8b949e",
    "text": "#e6edf3",
    "subtitle": "#8b949e",
}


def box(ax, x, y, w, h, color, label, sublabel=""):
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.05",
        facecolor=color + "33",
        edgecolor=color,
        linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2 + (0.15 if sublabel else 0),
        label,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=COLORS["text"],
    )
    if sublabel:
        ax.text(
            x + w / 2,
            y + h / 2 - 0.2,
            sublabel,
            ha="center",
            va="center",
            fontsize=7,
            color=COLORS["subtitle"],
        )


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.2),
    )


# Title
ax.text(
    8,
    8.5,
    "Energy-Seer — System Architecture",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color=COLORS["text"],
)

# Client
box(ax, 0.3, 6.5, 2.0, 1.0, COLORS["client"], "Clients", "HTTP / REST")

# FastAPI layer
box(
    ax,
    3.0,
    7.0,
    3.5,
    0.9,
    COLORS["api"],
    "FastAPI /api/v1/",
    "predict · anomaly · drift · forecast",
)
box(ax, 3.0, 5.8, 1.6, 0.9, COLORS["api"], "Rate Limiter", "300 req/min")
box(ax, 4.7, 5.8, 1.8, 0.9, COLORS["api"], "Correlation ID", "Middleware")

# ML Model
box(
    ax,
    7.5,
    6.5,
    3.5,
    1.4,
    COLORS["model"],
    "Ensemble Model",
    "XGBoost + LightGBM + RF\n5-fold CV · R² / RMSE",
)

# Feature Pipeline
box(
    ax,
    7.5,
    4.8,
    3.5,
    1.4,
    COLORS["model"],
    "Feature Pipeline",
    "Lag · Rolling · Temporal\nWeather Ratios · Building Enc.",
)

# Monitoring
box(
    ax,
    3.0,
    4.0,
    3.5,
    1.5,
    COLORS["monitoring"],
    "Monitoring",
    "KS-test Drift · IsolationForest\nZ-score Anomaly · Pred Logging",
)

# RAG / FAISS
box(
    ax,
    7.5,
    3.2,
    3.5,
    1.3,
    COLORS["rag"],
    "RAG / FAISS",
    "Pattern Similarity Search\nBaseline Consumption Lookup",
)

# Storage
box(ax, 11.5, 6.0, 2.8, 1.2, COLORS["storage"], "PostgreSQL", "SQLAlchemy ORM\nAlembic Migrations")
box(ax, 11.5, 4.5, 2.8, 1.2, COLORS["storage"], "Prediction Logs", "AnomalyLog · DriftLog")

# Airflow
box(
    ax,
    3.0,
    2.2,
    3.5,
    1.3,
    COLORS["pipeline"],
    "Airflow DAG",
    "Weekly Retrain · R² Gate\nRefresh Anomaly · Rebuild RAG",
)

# Docker
box(
    ax,
    7.5,
    1.5,
    7.8,
    0.9,
    COLORS["api"],
    "Docker + docker-compose",
    "API Container · PostgreSQL Container · Shared Volumes",
)

# Arrows
arrow(ax, 2.3, 7.0, 3.0, 7.4)
arrow(ax, 6.5, 7.4, 7.5, 7.2)
arrow(ax, 6.5, 7.0, 7.5, 5.5)
arrow(ax, 11.0, 7.2, 11.5, 6.6)
arrow(ax, 11.0, 5.5, 11.5, 5.1)
arrow(ax, 4.75, 5.8, 4.75, 5.5)
arrow(ax, 6.5, 4.7, 7.5, 3.9)
arrow(ax, 6.5, 4.2, 7.5, 3.6)
arrow(ax, 6.5, 2.8, 7.5, 2.1)

# Legend
legend_items = [
    mpatches.Patch(facecolor=COLORS["api"] + "55", edgecolor=COLORS["api"], label="API / Infra"),
    mpatches.Patch(
        facecolor=COLORS["model"] + "55", edgecolor=COLORS["model"], label="ML / Features"
    ),
    mpatches.Patch(
        facecolor=COLORS["monitoring"] + "55", edgecolor=COLORS["monitoring"], label="Monitoring"
    ),
    mpatches.Patch(
        facecolor=COLORS["storage"] + "55", edgecolor=COLORS["storage"], label="Storage"
    ),
    mpatches.Patch(facecolor=COLORS["rag"] + "55", edgecolor=COLORS["rag"], label="RAG / FAISS"),
    mpatches.Patch(
        facecolor=COLORS["pipeline"] + "55", edgecolor=COLORS["pipeline"], label="Airflow"
    ),
]
ax.legend(
    handles=legend_items,
    loc="lower left",
    fontsize=8,
    facecolor="#161b22",
    edgecolor="#30363d",
    labelcolor=COLORS["text"],
)

plt.tight_layout()
plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
print("Architecture diagram saved to screenshots/architecture.png")
