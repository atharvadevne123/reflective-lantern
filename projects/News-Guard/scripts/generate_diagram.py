"""Generate architecture diagram for News-Guard."""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

os.makedirs("screenshots", exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis("off")
fig.patch.set_facecolor("#f8f9fa")

def box(ax, x, y, w, h, color, text, fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.1",
        linewidth=1.5, edgecolor="#333", facecolor=color, zorder=2,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", wrap=True, zorder=3)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.5), zorder=1)

ax.text(8, 7.6, "News-Guard - System Architecture", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#222")

box(ax, 0.3, 5.5, 2.2, 1.0, "#aed6f1", "Client\n(REST / curl)", 9)
box(ax, 3.0, 5.5, 2.5, 1.0, "#a9dfbf", "FastAPI\n/predict /health /metrics", 8)
box(ax, 6.0, 6.2, 2.5, 0.8, "#f9e79f", "Feature Engineering\nTextExtractor + TF-IDF", 8)
box(ax, 6.0, 5.0, 2.5, 0.8, "#f0b27a", "Ensemble Model\nXGBoost + RF", 8)
box(ax, 9.5, 5.0, 2.5, 0.8, "#a9cce3", "KS-Drift Monitor\nreference vs current", 8)
box(ax, 6.5, 3.2, 2.5, 0.9, "#d5dbdb", "SQLAlchemy ORM\nPredictionLog + DriftLog", 8)
box(ax, 3.0, 3.2, 2.5, 0.9, "#fadbd8", "PostgreSQL\n(prod) / SQLite (dev)", 8)
box(ax, 12.5, 5.5, 2.8, 1.0, "#a8e6cf", "Airflow DAG\nWeekly Retrain", 8)
box(ax, 12.5, 3.2, 2.8, 0.9, "#ffd3b6", "GitHub Actions CI\nruff + pytest", 8)

arrow(ax, 2.5, 6.0, 3.0, 6.0)
arrow(ax, 5.5, 6.0, 6.0, 6.6)
arrow(ax, 5.5, 6.0, 6.0, 5.4)
arrow(ax, 8.5, 5.4, 9.5, 5.4)
arrow(ax, 7.0, 5.0, 7.0, 4.1)
arrow(ax, 6.5, 3.65, 5.5, 3.65)
arrow(ax, 12.0, 5.75, 12.5, 5.75)

ax.text(8, 0.4, "Python 3.11 | FastAPI | XGBoost + RF | sklearn Pipeline | SQLAlchemy | Docker | pytest",
        ha="center", va="center", fontsize=8, color="#555")

plt.tight_layout()
plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight")
print("Architecture diagram saved to screenshots/architecture.png")
