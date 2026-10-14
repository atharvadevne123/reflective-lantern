"""Generate system architecture diagram for Trade-Sage."""

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


def box(x, y, w, h, color, label, sublabel=""):
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.9,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2 + (0.15 if sublabel else 0),
        label,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="white",
    )
    if sublabel:
        ax.text(
            x + w / 2,
            y + h / 2 - 0.2,
            sublabel,
            ha="center",
            va="center",
            fontsize=6.5,
            color="#aaaaaa",
        )


def arrow(x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "color": "#58a6ff", "lw": 1.5},
    )


# Title
ax.text(
    8,
    8.5,
    "Trade-Sage — System Architecture",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color="white",
)

# Client
box(0.3, 6.5, 2.0, 1.0, "#21262d", "Client", "REST / HTTP")

# API layer
box(3.0, 6.5, 2.5, 1.0, "#1f6feb", "FastAPI", "/predict /portfolio\n/drift /health /metrics")

# Middleware
box(3.0, 5.0, 2.5, 1.0, "#388bfd", "Middleware", "RateLimit + CorrelationID\nGZip")

# Feature Engineering
box(6.5, 6.5, 2.5, 1.0, "#3fb950", "Feature Pipeline", "RSI MACD BB ATR\n22 indicators")

# ML Ensemble
box(10.0, 6.5, 2.5, 1.0, "#d29922", "ML Ensemble", "XGBoost + LightGBM\n+ RandomForest")

# Portfolio Risk
box(10.0, 4.8, 2.5, 1.0, "#bc8cff", "Portfolio Risk", "VaR ES Sharpe\nMax Drawdown")

# Drift Monitor
box(6.5, 4.8, 2.5, 1.0, "#f0883e", "Drift Monitor", "KS-test per feature\n22 columns")

# TTL Cache
box(13.5, 6.5, 2.2, 1.0, "#da3633", "TTL Cache", "300s LRU\n512 entries")

# Database
box(6.5, 3.0, 2.5, 1.0, "#388bfd", "PostgreSQL", "PredictionLog\nDriftLog ModelMetrics")

# Airflow DAG
box(3.0, 3.0, 2.5, 1.0, "#21262d", "Airflow DAG", "Champion/Challenger\nWeekly Retraining")

# Model Store
box(10.0, 3.0, 2.5, 1.0, "#21262d", "Model Store", "model.joblib\nmetrics.json")

# Arrows
arrow(2.3, 7.0, 3.0, 7.0)  # Client -> API
arrow(3.0, 7.0, 2.3, 7.0)  # API -> Client (response)
arrow(4.25, 6.5, 4.25, 6.0)  # API -> Middleware
arrow(4.25, 5.0, 4.25, 4.5)  # Middleware -> (down)
arrow(5.5, 7.0, 6.5, 7.0)  # API -> Features
arrow(9.0, 7.0, 10.0, 7.0)  # Features -> Ensemble
arrow(11.25, 6.5, 11.25, 5.8)  # Ensemble -> Portfolio
arrow(7.75, 6.5, 7.75, 5.8)  # Features -> Drift
arrow(12.5, 7.0, 13.5, 7.0)  # Ensemble -> Cache
arrow(7.75, 4.8, 7.75, 4.0)  # Drift -> DB
arrow(4.25, 3.0, 6.5, 3.5)  # Airflow -> DB
arrow(10.0, 3.5, 9.0, 3.5)  # Model Store <- Airflow (retrain)
arrow(4.25, 3.0, 5.5, 3.0)  # Airflow -> Model
arrow(5.5, 3.0, 10.0, 3.0)

# Legend
legend_items = [
    mpatches.Patch(color="#1f6feb", label="API Layer"),
    mpatches.Patch(color="#3fb950", label="Feature Engineering"),
    mpatches.Patch(color="#d29922", label="ML Ensemble"),
    mpatches.Patch(color="#f0883e", label="Monitoring"),
    mpatches.Patch(color="#bc8cff", label="Analytics"),
    mpatches.Patch(color="#21262d", label="Storage / Infra"),
]
ax.legend(
    handles=legend_items,
    loc="lower left",
    fontsize=7,
    facecolor="#161b22",
    edgecolor="#30363d",
    labelcolor="white",
)

plt.tight_layout(pad=0)
plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
print("Architecture diagram saved to screenshots/architecture.png")
