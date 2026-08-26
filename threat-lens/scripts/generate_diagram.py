"""Generate the Threat-Lens system architecture diagram."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# pad is in data units and inflates the box on every side — keep it small so
# the declared width/height are what actually gets drawn.
BOX_STYLE = "round,pad=0.03,rounding_size=0.18"
COLORS = {
    "ingress": "#2E5C8A",
    "compute": "#1F6F5C",
    "storage": "#7A4A8C",
    "monitor": "#B05A2A",
}


def _box(ax, x, y, w, h, label, color, fontsize=9.5):
    patch = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=BOX_STYLE,
        facecolor=color,
        edgecolor="white",
        linewidth=1.6,
        alpha=0.92,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center",
        color="white", fontsize=fontsize, fontweight="bold",
        linespacing=1.4,
    )


def _arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops={"arrowstyle": "-|>", "color": "#555555", "linewidth": 1.4},
    )
    if label:
        ax.text(
            (x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label,
            ha="center", va="bottom", fontsize=7.5, color="#444444",
        )


def generate() -> str:
    os.makedirs("screenshots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 7)
    ax.axis("off")
    plt.title(
        "Threat-Lens — System Architecture",
        fontsize=15, fontweight="bold", pad=16,
    )

    # Column 1 — ingress
    _box(ax, 0.3, 4.55, 2.4, 1.15, "Network Flow\n(NetFlow / IDS tap)", COLORS["ingress"])
    _box(ax, 0.3, 2.65, 2.4, 1.15, "REST Client\nPOST /api/v1/predict", COLORS["ingress"])

    # Column 2 — API
    _box(ax, 3.35, 2.65, 2.4, 3.05,
         "FastAPI\n\n/predict\n/health\n/metrics\n/drift\n/threats", COLORS["compute"])

    # Column 3 — feature pipeline, model, RAG
    _box(ax, 6.4, 5.05, 2.7, 1.05,
         "Feature Pipeline\n28 features · sklearn", COLORS["compute"])
    _box(ax, 6.4, 3.15, 2.7, 1.35,
         "Ensemble Model\nXGBoost + LightGBM\n+ RandomForest", COLORS["compute"])
    _box(ax, 6.4, 1.05, 2.7, 1.35,
         "Threat Intel RAG\nTF-IDF index\nCVE / MITRE ATT&CK", COLORS["compute"])

    # Column 4 — storage and MLOps
    _box(ax, 10.75, 4.75, 2.9, 1.35,
         "PostgreSQL\nprediction_logs\ndrift_reports", COLORS["storage"])
    _box(ax, 10.75, 2.9, 2.9, 1.2,
         "Drift Monitor\nKS-test (p < 0.05)", COLORS["monitor"])
    _box(ax, 10.75, 1.05, 2.9, 1.2,
         "Airflow DAG\nDaily retraining", COLORS["monitor"])

    # Ingress → API
    _arrow(ax, 2.7, 5.13, 3.35, 4.85)
    _arrow(ax, 2.7, 3.23, 3.35, 3.55)

    # API → feature pipeline → model
    _arrow(ax, 5.75, 4.95, 6.4, 5.45, "raw flow")
    _arrow(ax, 7.75, 5.05, 7.75, 4.5, "X")

    # API → model → RAG
    _arrow(ax, 5.75, 4.05, 6.4, 3.9, "predict")
    _arrow(ax, 5.75, 3.05, 6.4, 1.9, "attack ctx")

    # Model → storage
    _arrow(ax, 9.1, 4.0, 10.75, 5.1, "log")

    # Storage → drift monitor → Airflow → model
    _arrow(ax, 12.2, 4.75, 12.2, 4.1, "sample")
    _arrow(ax, 12.2, 2.9, 12.2, 2.25, "drift")
    _arrow(ax, 10.75, 1.65, 9.1, 3.35, "new model")

    legend = [
        mpatches.Patch(color=COLORS["ingress"], label="Ingress"),
        mpatches.Patch(color=COLORS["compute"], label="Compute / ML"),
        mpatches.Patch(color=COLORS["storage"], label="Storage"),
        mpatches.Patch(color=COLORS["monitor"], label="Monitoring / MLOps"),
    ]
    ax.legend(
        handles=legend, loc="lower left",
        bbox_to_anchor=(0.0, -0.04), ncol=4, frameon=False, fontsize=9,
    )

    out = "screenshots/architecture.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(f"Diagram written to {generate()}")
