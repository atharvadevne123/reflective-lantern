"""Generate Cyber-Guard system architecture diagram."""

from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

os.makedirs("screenshots", exist_ok=True)


def draw_box(ax, x, y, w, h, label, color="#2563eb", fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
        alpha=0.88,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white", wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate(
        label, xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color="#374151", lw=1.5),
        fontsize=7.5, color="#374151", ha="center",
    )


def main():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")

    plt.title(
        "Cyber-Guard — System Architecture",
        fontsize=15, fontweight="bold", color="#0f172a", pad=15,
    )

    # Client
    draw_box(ax, 0.3, 3.5, 1.8, 1.0, "Client /\nSOC Analyst", color="#1d4ed8")

    # FastAPI
    draw_box(ax, 2.8, 2.8, 2.2, 2.4, "FastAPI\n/predict\n/health\n/metrics\n/drift", color="#0891b2")

    # Feature Engineering
    draw_box(ax, 6.0, 5.0, 2.2, 1.2, "Feature\nEngineering\nPipeline", color="#7c3aed")

    # ML Ensemble
    draw_box(ax, 6.0, 3.2, 2.2, 1.6, "ML Ensemble\nXGBoost\nRandomForest", color="#059669")

    # Monitoring
    draw_box(ax, 6.0, 1.2, 2.2, 1.6, "KS-Drift\nMonitor", color="#d97706")

    # PostgreSQL
    draw_box(ax, 9.5, 3.5, 2.0, 1.2, "PostgreSQL\n(predictions\n+ drift logs)", color="#dc2626")

    # Airflow DAG
    draw_box(ax, 9.5, 1.2, 2.0, 1.6, "Airflow DAG\nWeekly\nRetrain", color="#6d28d9")

    # Model Store
    draw_box(ax, 12.5, 3.5, 2.0, 1.2, "Model Store\nmodel.joblib\nmetrics.json", color="#0f766e")

    # Arrows
    draw_arrow(ax, 2.1, 4.0, 2.8, 4.0, "HTTP")
    draw_arrow(ax, 5.0, 5.2, 6.0, 5.6, "raw feats")
    draw_arrow(ax, 5.0, 4.0, 6.0, 4.0, "predict")
    draw_arrow(ax, 5.0, 3.2, 6.0, 1.8, "log")
    draw_arrow(ax, 8.2, 5.2, 9.5, 4.2, "features")
    draw_arrow(ax, 8.2, 3.8, 9.5, 3.9, "store")
    draw_arrow(ax, 8.2, 1.8, 9.5, 1.8, "drift log")
    draw_arrow(ax, 11.5, 2.0, 12.5, 3.8, "retrain →\nmodel.joblib")
    draw_arrow(ax, 11.5, 4.0, 12.5, 4.0, "read/write")

    # Legend
    legend_items = [
        mpatches.Patch(color="#0891b2", label="API Layer"),
        mpatches.Patch(color="#7c3aed", label="Feature Engineering"),
        mpatches.Patch(color="#059669", label="ML Ensemble"),
        mpatches.Patch(color="#d97706", label="Drift Monitor"),
        mpatches.Patch(color="#dc2626", label="Database"),
        mpatches.Patch(color="#6d28d9", label="Retraining Pipeline"),
        mpatches.Patch(color="#0f766e", label="Model Store"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out_path = "screenshots/architecture.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Architecture diagram saved: {out_path}")


if __name__ == "__main__":
    main()
