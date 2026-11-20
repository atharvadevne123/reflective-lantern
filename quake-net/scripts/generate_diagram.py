"""Render the Quake-Net system architecture diagram to screenshots/architecture.png."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1b2430"
MUTED = "#5a6b7d"

LAYERS = [
    ("#e8f1fb", "#2f6fb0"),  # ingress
    ("#eaf6ee", "#2f8f57"),  # ml core
    ("#fdf0e6", "#c2701c"),  # monitoring
    ("#f1ecfa", "#6b4fbb"),  # storage / ops
]

BOXES = [
    # (x, y, w, h, title, subtitle, layer)
    (0.4, 5.5, 2.6, 1.0, "Seismic Feed", "stations / REST clients", 0),
    (0.4, 4.1, 2.6, 1.0, "FastAPI Gateway", "rate limit - correlation ID", 0),
    (3.6, 5.5, 3.0, 1.0, "Feature Pipeline", "6 stages - 40+ features", 1),
    (3.6, 4.1, 3.0, 1.0, "Voting Ensemble", "XGBoost 0.6 + RandomForest 0.4", 1),
    (3.6, 2.7, 3.0, 1.0, "Aftershock Head", "logistic on magnitude", 1),
    (7.2, 5.5, 3.0, 1.0, "Drift Monitor", "KS-test + PSI", 2),
    (7.2, 4.1, 3.0, 1.0, "Anomaly Detector", "IsolationForest + z / IQR", 2),
    (7.2, 2.7, 3.0, 1.0, "Similarity Index", "FAISS + brute-force fallback", 2),
    (10.8, 5.5, 2.8, 1.0, "PostgreSQL", "events - drift - metrics", 3),
    (10.8, 4.1, 2.8, 1.0, "Retrain DAG", "champion / challenger", 3),
    (10.8, 2.7, 2.8, 1.0, "TTL Cache", "LRU - 300s", 3),
]

ARROWS = [
    ((3.0, 6.0), (3.6, 6.0)),
    ((1.7, 5.5), (1.7, 5.1)),
    ((3.0, 4.6), (3.6, 4.6)),
    ((5.1, 5.5), (5.1, 5.1)),
    ((5.1, 4.1), (5.1, 3.7)),
    ((6.6, 4.6), (7.2, 4.6)),
    ((6.6, 6.0), (7.2, 6.0)),
    ((6.6, 3.2), (7.2, 3.2)),
    ((10.2, 6.0), (10.8, 6.0)),
    ((10.2, 4.6), (10.8, 4.6)),
    ((10.2, 3.2), (10.8, 3.2)),
    ((12.2, 4.1), (12.2, 3.7)),
]


def build_figure() -> plt.Figure:
    """Compose the architecture figure without writing it to disk."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(1.6, 7.4)
    ax.axis("off")

    for x, y, w, h, title, subtitle, layer in BOXES:
        face, edge = LAYERS[layer]
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.06,rounding_size=0.12",
                facecolor=face,
                edgecolor=edge,
                linewidth=1.6,
            )
        )
        ax.text(
            x + w / 2,
            y + h * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=INK,
        )
        ax.text(
            x + w / 2, y + h * 0.26, subtitle, ha="center", va="center", fontsize=8, color=MUTED
        )

    for (x1, y1), (x2, y2) in ARROWS:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 1.4},
        )

    ax.text(0.4, 7.15, "Quake-Net - System Architecture", fontsize=16, fontweight="bold", color=INK)
    ax.text(
        0.4,
        6.82,
        "Seismic magnitude prediction, aftershock forecasting and drift monitoring",
        fontsize=9.5,
        color=MUTED,
    )

    legend = [
        mpatches.Patch(facecolor=LAYERS[0][0], edgecolor=LAYERS[0][1], label="Ingress"),
        mpatches.Patch(facecolor=LAYERS[1][0], edgecolor=LAYERS[1][1], label="ML core"),
        mpatches.Patch(facecolor=LAYERS[2][0], edgecolor=LAYERS[2][1], label="Monitoring"),
        mpatches.Patch(facecolor=LAYERS[3][0], edgecolor=LAYERS[3][1], label="Storage & ops"),
    ]
    ax.legend(
        handles=legend,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.10),
        fontsize=9,
    )

    fig.tight_layout()
    return fig


def main(output: str = "screenshots/architecture.png") -> str:
    """Render the diagram to ``output`` and return the path."""
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig = build_figure()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Architecture diagram written to {output}")
    return output


if __name__ == "__main__":
    main()
