"""Generate the system architecture diagram for Support-Sage."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def generate(output_path: str = "screenshots/architecture.png") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    def box(x, y, w, h, label, sublabel="", color="#1e40af", fontsize=9):
        rect = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1", facecolor=color, edgecolor="#60a5fa", linewidth=1.5
        )
        ax.add_patch(rect)
        if sublabel:
            ax.text(x + w / 2, y + h * 0.65, label, ha="center", va="center", fontsize=fontsize, color="white", fontweight="bold")
            ax.text(x + w / 2, y + h * 0.3, sublabel, ha="center", va="center", fontsize=7, color="#93c5fd", style="italic")
        else:
            ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, color="white", fontweight="bold")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#60a5fa", lw=1.5))

    plt.title("Support-Sage - System Architecture", fontsize=16, fontweight="bold", color="white", pad=15)
    box(0.3, 4.0, 1.8, 1.2, "Client", "HTTP REST", "#7c3aed")
    box(2.5, 4.0, 2.2, 1.2, "FastAPI", "/predict /health\n/metrics /train", "#1e40af")
    box(5.2, 5.5, 2.4, 1.2, "Feature Pipeline", "TF-IDF + Meta (5024)", "#065f46")
    box(5.2, 3.8, 2.4, 1.2, "ML Ensemble", "XGBoost+LightGBM+RF", "#065f46")
    box(5.2, 2.2, 2.4, 1.0, "LightGBM Reg", "Resolution Hours", "#065f46")
    box(8.2, 5.5, 2.4, 1.2, "FAISS Index", "RAG 128-dim SVD", "#7c2d12")
    box(8.2, 3.8, 2.4, 1.2, "KS-Drift Monitor", "Confidence / Hours / WC", "#7c2d12")
    box(11.0, 4.5, 2.2, 1.0, "PostgreSQL", "Prod Storage", "#1e3a5f")
    box(11.0, 3.2, 2.2, 1.0, "SQLite", "Dev Storage", "#374151")
    box(11.0, 1.5, 2.2, 1.2, "Airflow DAG", "Weekly Retrain", "#713f12")
    box(8.2, 1.5, 2.4, 1.2, "GitHub Actions", "ruff + pytest CI", "#1a1a2e")
    arrow(2.1, 4.6, 2.5, 4.6)
    arrow(4.7, 5.2, 5.2, 6.1)
    arrow(4.7, 4.6, 5.2, 4.4)
    arrow(4.7, 4.3, 5.2, 2.7)
    arrow(7.6, 6.1, 8.2, 6.1)
    arrow(7.6, 4.4, 8.2, 4.4)
    arrow(10.6, 4.4, 11.0, 5.0)
    arrow(10.6, 4.4, 11.0, 3.7)
    arrow(13.2, 3.7, 13.2, 2.7)
    arrow(7.6, 2.0, 8.2, 2.0)
    legend_items = [
        mpatches.Patch(color="#1e40af", label="API Layer"),
        mpatches.Patch(color="#065f46", label="ML Models"),
        mpatches.Patch(color="#7c2d12", label="Monitoring/RAG"),
        mpatches.Patch(color="#1e3a5f", label="Storage"),
        mpatches.Patch(color="#713f12", label="Orchestration"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8, facecolor="#1e293b", edgecolor="#60a5fa", labelcolor="white")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    generate()
