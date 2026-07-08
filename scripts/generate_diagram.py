"""Generate Realty-Edge system architecture diagram."""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

os.makedirs("screenshots", exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")

COLORS = {
    "client": "#4A90D9",
    "api": "#2ECC71",
    "ml": "#E74C3C",
    "storage": "#F39C12",
    "pipeline": "#9B59B6",
    "monitoring": "#1ABC9C",
    "header": "#2C3E50",
}


def box(ax, x, y, w, h, color, label, sublabel="", fontsize=10):
    rect = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
        alpha=0.9,
    )
    ax.add_patch(rect)
    cy = y + h / 2
    ax.text(
        x + w / 2,
        cy + (0.1 if sublabel else 0),
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="white",
    )
    if sublabel:
        ax.text(
            x + w / 2,
            cy - 0.22,
            sublabel,
            ha="center",
            va="center",
            fontsize=7.5,
            color="white",
            alpha=0.9,
        )


def arrow(ax, x1, y1, x2, y2, color="#7F8C8D"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5}
    )


ax.text(
    8,
    8.5,
    "Realty-Edge — System Architecture",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    color=COLORS["header"],
)

box(ax, 0.3, 6.8, 2.0, 0.9, COLORS["client"], "Client", "REST / JSON")
box(ax, 0.3, 5.5, 2.0, 0.9, COLORS["client"], "Batch Client", "CSV / bulk")

box(ax, 3.2, 7.0, 2.8, 0.7, COLORS["api"], "/api/v1/predict", "POST")
box(ax, 3.2, 6.1, 2.8, 0.7, COLORS["api"], "/api/v1/batch-predict", "POST")
box(ax, 3.2, 5.2, 2.8, 0.7, COLORS["api"], "/api/v1/comparable-properties", "POST")
box(ax, 3.2, 4.3, 2.8, 0.7, COLORS["api"], "/api/v1/neighborhood-stats", "GET")
box(ax, 3.2, 3.4, 2.8, 0.7, COLORS["api"], "/api/v1/drift-status", "GET")

box(ax, 7.2, 6.8, 2.5, 0.9, COLORS["ml"], "XGBoost", "n_est=200")
box(ax, 7.2, 5.7, 2.5, 0.9, COLORS["ml"], "LightGBM", "n_est=200")
box(ax, 7.2, 4.6, 2.5, 0.9, COLORS["ml"], "RandomForest", "n_est=100")
box(ax, 7.2, 3.5, 2.5, 0.9, COLORS["ml"], "VotingRegressor", "soft ensemble")

box(ax, 10.5, 7.1, 2.5, 0.7, COLORS["storage"], "PostgreSQL", "prod storage")
box(ax, 10.5, 6.0, 2.5, 0.7, COLORS["storage"], "SQLite", "dev storage")
box(ax, 10.5, 4.9, 2.5, 0.7, COLORS["monitoring"], "FAISS Index", "comparable search")
box(ax, 10.5, 3.8, 2.5, 0.7, COLORS["pipeline"], "Airflow DAG", "weekly retrain")

box(ax, 13.2, 6.5, 2.5, 0.7, COLORS["monitoring"], "KS-test", "drift detection")
box(ax, 13.2, 5.4, 2.5, 0.7, COLORS["monitoring"], "PredictionLog", "DB logging")
box(ax, 13.2, 4.3, 2.5, 0.7, COLORS["monitoring"], "Rate Limiter", "300 req/min")
box(ax, 13.2, 3.2, 2.5, 0.7, COLORS["pipeline"], "5-fold CV", "R2+RMSE metrics")

ax.text(4.6, 8.2, "FastAPI Layer", ha="center", fontsize=9, color=COLORS["api"], fontstyle="italic")
ax.text(8.45, 8.2, "ML Ensemble", ha="center", fontsize=9, color=COLORS["ml"], fontstyle="italic")
ax.text(
    11.75,
    8.2,
    "Storage & Search",
    ha="center",
    fontsize=9,
    color=COLORS["storage"],
    fontstyle="italic",
)
ax.text(
    14.45,
    8.2,
    "Monitoring",
    ha="center",
    fontsize=9,
    color=COLORS["monitoring"],
    fontstyle="italic",
)

for y in [7.25, 5.9]:
    arrow(ax, 2.3, y, 3.2, y)
for y in [7.35, 6.45, 5.55, 4.65]:
    arrow(ax, 6.0, y, 7.2, y)
for y in [7.45, 6.35, 5.25]:
    arrow(ax, 9.7, y, 10.5, y)
for y in [6.8, 5.65, 4.55, 3.45]:
    arrow(ax, 13.0, y, 13.2, y)

stack_items = [
    "Python 3.11",
    "FastAPI",
    "XGBoost+LightGBM+RF",
    "sklearn Pipeline",
    "FAISS Search",
    "KS-drift Monitoring",
    "PostgreSQL+SQLAlchemy",
    "Docker+Compose",
    "Airflow DAG",
    "GitHub Actions CI",
]
ax.text(0.3, 3.0, "Tech Stack:", fontsize=9, fontweight="bold", color=COLORS["header"])
for i, item in enumerate(stack_items):
    col = i % 2
    row = i // 2
    ax.text(0.3 + col * 3.8, 2.6 - row * 0.35, f"• {item}", fontsize=8.0, color="#34495E")

plt.tight_layout()
plt.savefig(
    "screenshots/architecture.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
)
print("Architecture diagram saved to screenshots/architecture.png")
