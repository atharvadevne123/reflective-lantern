"""Generate the Forge-Guard system architecture diagram."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.makedirs("screenshots", exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

COLORS = {
    "sensor": "#1e88e5",
    "feature": "#43a047",
    "model": "#e53935",
    "api": "#fb8c00",
    "db": "#8e24aa",
    "monitor": "#00acc1",
    "retrain": "#6d4c41",
    "client": "#546e7a",
    "header": "#eceff1",
    "arrow": "#90a4ae",
}


def box(ax, x, y, w, h, color, label, sublabel="", fontsize=9):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.88,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2 + (0.15 if sublabel else 0),
        label, ha="center", va="center",
        fontsize=fontsize, fontweight="bold", color="white", zorder=4,
    )
    if sublabel:
        ax.text(
            x + w / 2, y + h / 2 - 0.22,
            sublabel, ha="center", va="center",
            fontsize=7, color="#cfd8dc", zorder=4,
        )


def arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "color": COLORS["arrow"], "lw": 1.5},
        zorder=2,
    )


# Title
ax.text(8, 8.5, "Forge-Guard — System Architecture", ha="center", va="center",
        fontsize=16, fontweight="bold", color=COLORS["header"])

# Layer labels
for y_pos, label in [(7.6, "INGESTION"), (6.0, "FEATURE ENGINEERING"), (4.4, "INFERENCE"),
                      (2.8, "PERSISTENCE"), (1.2, "OPS")]:
    ax.text(0.3, y_pos, label, ha="left", va="center", fontsize=6.5,
            color="#78909c", fontweight="bold")

# Row 1 — sensors
for i, (name, sub) in enumerate([
    ("Temperature", "°C"), ("Pressure", "bar"),
    ("Vibration", "mm/s"), ("Tool Wear", "µm"),
    ("Power", "kW"), ("Humidity", "%"),
]):
    box(ax, 1.0 + i * 2.3, 7.1, 1.9, 0.8, COLORS["sensor"], name, sub, fontsize=8)
    arrow(ax, 1.95 + i * 2.3, 7.1, 3.0 + i * 0.65, 6.5)

# Row 2 — feature pipeline
for i, (name, sub) in enumerate([
    ("Lag (t-1,t-2)", "7×2 cols"), ("Rolling μ/σ", "7×2 cols"),
    ("Ratios", "4 cols"), ("Poly (x²)", "3 cols"),
    ("StandardScaler", "normalise"),
]):
    box(ax, 0.5 + i * 3.0, 5.5, 2.6, 0.8, COLORS["feature"], name, sub, fontsize=8)

arrow(ax, 1.8, 5.5, 1.8, 5.0)
arrow(ax, 4.8, 5.5, 4.8, 5.0)
arrow(ax, 7.8, 5.5, 7.8, 5.0)
arrow(ax, 10.8, 5.5, 10.8, 5.0)
arrow(ax, 13.8, 5.5, 13.8, 5.0)

# Row 3 — model ensemble + API
box(ax, 3.5, 3.8, 3.2, 1.0, COLORS["model"], "XGBoost", "n=200, depth=5", fontsize=9)
box(ax, 7.2, 3.8, 3.2, 1.0, COLORS["model"], "RandomForest", "n=150, depth=7", fontsize=9)
box(ax, 5.3, 3.8, 1.6, 1.0, COLORS["model"], "Soft Vote", "ensemble", fontsize=8)
arrow(ax, 5.1, 3.8, 5.9, 3.8)
arrow(ax, 8.8, 3.8, 6.9, 3.8)

box(ax, 11.5, 3.8, 3.8, 1.0, COLORS["api"], "FastAPI", "/api/v1/predict\n/health  /metrics", fontsize=8)
arrow(ax, 6.9, 4.3, 11.5, 4.3)

# Client arrow
box(ax, 13.5, 6.8, 2.0, 0.8, COLORS["client"], "Client", "REST JSON", fontsize=8)
arrow(ax, 14.5, 6.8, 13.9, 4.8)
arrow(ax, 13.9, 4.8, 14.5, 5.6)

# Row 4 — DB + monitoring
box(ax, 1.0, 2.3, 3.5, 1.0, COLORS["db"], "PostgreSQL", "prediction_logs\ndrift_reports", fontsize=8)
box(ax, 5.5, 2.3, 3.5, 1.0, COLORS["monitor"], "Drift Monitor", "KS-test per feature\np < 0.05 → alert", fontsize=8)
box(ax, 10.0, 2.3, 3.5, 1.0, COLORS["db"], "SQLAlchemy ORM", "SQLite dev\nPostgres prod", fontsize=8)

arrow(ax, 13.3, 3.8, 13.3, 3.3)
arrow(ax, 13.3, 3.3, 11.5, 3.3)
arrow(ax, 7.2, 2.8, 5.5, 2.8)

# Row 5 — retraining + CI
box(ax, 1.0, 0.8, 3.5, 1.0, COLORS["retrain"], "Retrain Pipeline", "extract→engineer→train\nrecord AUC delta", fontsize=8)
box(ax, 5.5, 0.8, 3.5, 1.0, COLORS["retrain"], "GitHub Actions CI", "ruff lint + pytest\nPy 3.10/3.11/3.12", fontsize=8)
box(ax, 10.0, 0.8, 3.5, 1.0, COLORS["monitor"], "Docker Compose", "API + PostgreSQL\nhealthcheck", fontsize=8)

arrow(ax, 2.75, 2.3, 2.75, 1.8)
arrow(ax, 7.25, 2.3, 7.25, 1.8)

plt.tight_layout(pad=0.3)
plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Architecture diagram saved to screenshots/architecture.png")
