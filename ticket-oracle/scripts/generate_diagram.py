"""Generate Ticket-Oracle system architecture diagram."""

from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_box(ax, x, y, w, h, label, color="#2563EB", fontsize=9):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center",
        fontsize=fontsize, color="white", fontweight="bold", zorder=4,
    )


def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "color": "#94A3B8", "lw": 1.5},
        zorder=2,
    )


def main():
    os.makedirs("screenshots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(
        8, 8.5, "Ticket-Oracle — System Architecture",
        ha="center", va="center", fontsize=14, color="white", fontweight="bold",
    )

    # Client
    draw_box(ax, 0.3, 7.0, 2.2, 0.8, "Client / Helpdesk\nUI", color="#475569")

    # API layer
    draw_box(ax, 3.5, 6.8, 3.0, 1.2, "FastAPI\n/api/v1/*\n(rate-limit, CID)", color="#2563EB")

    # Feature engineering
    draw_box(ax, 7.5, 7.0, 2.5, 0.8, "Feature Pipeline\nTF-IDF + OrdinalEnc\n+ StandardScaler", color="#7C3AED")

    # Models
    draw_box(ax, 7.5, 5.5, 1.1, 1.2, "Priority\nModel\nXGB+LGBM\n+RF", color="#059669")
    draw_box(ax, 9.0, 5.5, 1.0, 1.2, "SLA Breach\nModel\nXGBoost", color="#059669")

    # Monitoring
    draw_box(ax, 7.5, 4.0, 2.5, 0.9, "Drift Monitor\nKS-test + PSI\nPrediction Log", color="#D97706")

    # Database
    draw_box(ax, 3.5, 4.0, 2.5, 1.2, "SQLAlchemy ORM\nPredictionLog\nDriftLog\nModelMetrics", color="#1E40AF")

    # PostgreSQL
    draw_box(ax, 3.5, 2.3, 2.5, 0.9, "PostgreSQL\n(Docker)", color="#1E3A5F")

    # Airflow DAG
    draw_box(ax, 7.5, 2.3, 2.5, 1.2, "Airflow DAG\nWeekly Retrain\nChampion/Challenger\nAUC Gate", color="#9333EA")

    # Model store
    draw_box(ax, 11.5, 4.0, 2.0, 0.8, "Model Store\n(joblib)\nmetrics.json", color="#374151")

    # CI/CD
    draw_box(ax, 11.5, 5.5, 2.0, 1.2, "GitHub Actions CI\nruff lint\npytest 3.11/3.12\nwheel build", color="#64748B")

    # Arrows
    draw_arrow(ax, 2.5, 7.4, 3.5, 7.3)
    draw_arrow(ax, 6.5, 7.3, 7.5, 7.4)
    draw_arrow(ax, 8.75, 7.0, 8.25, 6.7)
    draw_arrow(ax, 9.25, 7.0, 9.5, 6.7)
    draw_arrow(ax, 8.0, 5.5, 7.8, 4.9)
    draw_arrow(ax, 9.5, 5.5, 9.0, 4.9)
    draw_arrow(ax, 7.5, 4.45, 6.0, 4.6)
    draw_arrow(ax, 5.0, 4.0, 5.0, 3.2)
    draw_arrow(ax, 8.75, 4.0, 8.75, 3.5)
    draw_arrow(ax, 10.0, 6.1, 11.5, 6.1)
    draw_arrow(ax, 10.0, 4.45, 11.5, 4.45)

    # Legend
    legend_items = [
        mpatches.Patch(color="#2563EB", label="API Layer"),
        mpatches.Patch(color="#7C3AED", label="Feature Engineering"),
        mpatches.Patch(color="#059669", label="ML Models"),
        mpatches.Patch(color="#D97706", label="Monitoring"),
        mpatches.Patch(color="#9333EA", label="Retraining Pipeline"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8, facecolor="#1E293B",
              labelcolor="white", framealpha=0.8)

    plt.tight_layout()
    out = "screenshots/architecture.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
