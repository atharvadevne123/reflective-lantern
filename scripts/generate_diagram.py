"""Generate Volt-Cast system architecture diagram."""

from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

os.makedirs("screenshots", exist_ok=True)


def draw_box(ax, x, y, w, h, label, color, fontsize=9, textcolor="white"):
    box = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=textcolor, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.5},
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.05, my, label, fontsize=7, color="#333333")


fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
ax.set_facecolor("#F8F9FA")
fig.patch.set_facecolor("#F8F9FA")

plt.title("Volt-Cast — System Architecture", fontsize=16, fontweight="bold", pad=20, color="#1A1A2E")

# Colors
CLIENT_C = "#4361EE"
API_C = "#3A0CA3"
ML_C = "#7209B7"
DB_C = "#F72585"
MONITOR_C = "#4CC9F0"
PIPE_C = "#06D6A0"
FAISS_C = "#FFB703"

# Client
draw_box(ax, 0.3, 6.5, 2.2, 1.2, "API Clients\n(REST / Batch)", CLIENT_C)

# FastAPI Layer
draw_box(ax, 3.5, 5.0, 2.8, 3.0, "FastAPI\n/predict\n/batch-predict\n/forecast\n/drift\n/retrain\n/health /metrics", API_C, fontsize=7.5)

# ML Ensemble
draw_box(ax, 7.2, 5.8, 2.8, 2.0, "ML Ensemble\nXGBoost\nLightGBM\nRandomForest\nVotingRegressor", ML_C, fontsize=8)

# Feature Pipeline
draw_box(ax, 7.2, 3.5, 2.8, 1.8, "Feature Pipeline\nLag (1h/24h/168h)\nRolling Mean/Std\nTemporal Cyclical\nPeak Encoder\nRatio Features", ML_C, fontsize=7.5)

# Database
draw_box(ax, 11.0, 5.8, 2.8, 2.0, "PostgreSQL\nEnergyReadings\nPredictionLog\nDriftReports\nRetrainingLog", DB_C, fontsize=8)

# Monitoring
draw_box(ax, 11.0, 3.5, 2.8, 1.8, "Monitoring\nKS-Test Drift\nRMSE/MAE/MAPE\nP50/P95 Latency\nPredictionTracker", MONITOR_C, fontsize=7.5, textcolor="#1A1A2E")

# Airflow
draw_box(ax, 7.2, 1.2, 2.8, 1.8, "Airflow DAG\nWeekly Retrain\nData Gate\nDrift Alert\nR² Validation", PIPE_C, fontsize=8, textcolor="#1A1A2E")

# Docker
draw_box(ax, 3.5, 1.2, 2.8, 1.8, "Docker Compose\nAPI Container\nPostgreSQL\n.env.example\nGH Actions CI", FAISS_C, fontsize=8, textcolor="#1A1A2E")

# Arrows
draw_arrow(ax, 2.5, 7.1, 3.5, 7.1, "REST")
draw_arrow(ax, 6.3, 6.8, 7.2, 6.8, "predict()")
draw_arrow(ax, 6.3, 6.2, 7.2, 4.5, "features")
draw_arrow(ax, 10.0, 6.8, 11.0, 6.8, "DB write")
draw_arrow(ax, 10.0, 4.3, 11.0, 4.3, "drift write")
draw_arrow(ax, 8.6, 3.5, 8.6, 3.0, "schedule")
draw_arrow(ax, 7.2, 2.1, 6.3, 2.1, "trigger")
draw_arrow(ax, 10.0, 2.1, 11.0, 4.3, "log retrain")

legend_elements = [
    mpatches.Patch(facecolor=CLIENT_C, label="Client"),
    mpatches.Patch(facecolor=API_C, label="FastAPI"),
    mpatches.Patch(facecolor=ML_C, label="ML / Features"),
    mpatches.Patch(facecolor=DB_C, label="PostgreSQL"),
    mpatches.Patch(facecolor=MONITOR_C, label="Monitoring"),
    mpatches.Patch(facecolor=PIPE_C, label="Airflow Pipeline"),
    mpatches.Patch(facecolor=FAISS_C, label="Infra / Docker"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.8)

plt.tight_layout()
plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight")
print("Architecture diagram saved to screenshots/architecture.png")
