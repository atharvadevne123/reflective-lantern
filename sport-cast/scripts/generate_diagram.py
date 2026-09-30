"""Generate Sport-Cast system architecture diagram."""
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

os.makedirs("screenshots", exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis("off")
fig.patch.set_facecolor("#0d1117")

colors = {
    "client": "#1f6feb",
    "api": "#238636",
    "ml": "#9e6a03",
    "db": "#6e40c9",
    "monitor": "#da3633",
    "pipe": "#388bfd",
}


def box(ax, x, y, w, h, label, color, fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="white", linewidth=1.2, alpha=0.85,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            color="white", fontsize=fontsize, fontweight="bold", wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops={"arrowstyle": "->", "color": "#58a6ff", "lw": 1.5})


ax.text(8, 7.6, "Sport-Cast — System Architecture", ha="center", va="center",
        color="white", fontsize=14, fontweight="bold")

box(ax, 0.3, 5.5, 2.0, 1.0, "Client /\nAPI Consumer", colors["client"])
box(ax, 3.0, 5.5, 2.5, 1.0, "FastAPI\n/api/v1/predict\n/health /metrics", colors["api"])
box(ax, 6.5, 5.5, 2.5, 1.0, "Feature Pipeline\n(Form / Lag / H2H\nRatio / Fatigue)", colors["pipe"])
box(ax, 10.0, 5.5, 2.5, 1.0, "Ensemble Model\nXGBoost + LightGBM\n+ RandomForest", colors["ml"])
box(ax, 3.0, 3.5, 2.5, 1.0, "Monitoring\nKS-test drift\nPrediction log", colors["monitor"])
box(ax, 6.5, 3.5, 2.5, 1.0, "SQLAlchemy ORM\nPredictionLog\nDriftLog", colors["db"])
box(ax, 10.0, 3.5, 2.5, 1.0, "PostgreSQL\n(prod)\nSQLite (dev)", colors["db"])
box(ax, 6.5, 1.5, 2.5, 1.0, "Airflow DAG\nWeekly retrain\nChampion/Challenger", colors["pipe"])
box(ax, 10.0, 1.5, 2.5, 1.0, "Player Scoring\nGoals / Win-rate\nFatigue index", colors["ml"])

arrow(ax, 2.3, 6.0, 3.0, 6.0)
arrow(ax, 5.5, 6.0, 6.5, 6.0)
arrow(ax, 9.0, 6.0, 10.0, 6.0)
arrow(ax, 4.25, 5.5, 4.25, 4.5)
arrow(ax, 7.75, 5.5, 7.75, 4.5)
arrow(ax, 9.0, 4.0, 10.0, 4.0)
arrow(ax, 7.75, 3.5, 7.75, 2.5)

legend_items = [
    mpatches.Patch(color=colors["client"], label="External"),
    mpatches.Patch(color=colors["api"], label="API Layer"),
    mpatches.Patch(color=colors["pipe"], label="Pipelines"),
    mpatches.Patch(color=colors["ml"], label="ML / Scoring"),
    mpatches.Patch(color=colors["db"], label="Storage"),
    mpatches.Patch(color=colors["monitor"], label="Monitoring"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=8,
          facecolor="#161b22", labelcolor="white", edgecolor="#30363d")

plt.tight_layout()
plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Architecture diagram saved to screenshots/architecture.png")
