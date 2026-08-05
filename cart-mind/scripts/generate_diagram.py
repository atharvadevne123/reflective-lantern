"""Generate Cart-Mind system architecture diagram."""

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

os.makedirs("screenshots", exist_ok=True)

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis("off")
plt.title("Cart-Mind — System Architecture", fontsize=14, fontweight="bold")

boxes = [
    (0.3, 5.5, 2.0, 0.8, "#4CAF50", "Client\n(REST)"),
    (3.0, 5.5, 2.5, 0.8, "#2196F3", "FastAPI\n/api/v1/*"),
    (6.2, 6.2, 2.2, 0.8, "#FF9800", "LightGBM+XGB+RF\nEnsemble"),
    (6.2, 5.1, 2.2, 0.8, "#9C27B0", "FAISS\nItem Index"),
    (6.2, 4.0, 2.2, 0.8, "#F44336", "Feature Pipeline\n5-stage sklearn"),
    (9.5, 5.5, 2.2, 0.8, "#607D8B", "PostgreSQL\nPrediction Logs"),
    (9.5, 4.2, 2.2, 0.8, "#795548", "Drift Monitor\nKS-test"),
    (3.0, 3.2, 2.5, 0.8, "#00BCD4", "Airflow DAG\nWeekly Retrain"),
]

for x, y, w, h, color, label in boxes:
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="white",
            linewidth=1.5,
            alpha=0.85,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        fontweight="bold",
    )

arrows = [
    (2.3, 5.9, 0.65, 0),
    (5.5, 5.9, 0.65, 0.35),
    (5.5, 5.9, 0.65, -0.4),
    (5.5, 5.9, 0.65, -1.5),
    (8.4, 5.9, 1.05, 0),
    (8.4, 4.6, 1.05, 0),
]
for sx, sy, dx, dy in arrows:
    ax.annotate(
        "",
        xy=(sx + dx, sy + dy),
        xytext=(sx, sy),
        arrowprops=dict(arrowstyle="->", color="#333333", lw=1.4),
    )

ax.text(
    7.0,
    1.5,
    "Tech: FastAPI · LightGBM · XGBoost · FAISS · PostgreSQL · Docker · Airflow",
    ha="center",
    fontsize=8,
    color="#555555",
    style="italic",
)

plt.tight_layout()
plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight")
print("Saved screenshots/architecture.png")
