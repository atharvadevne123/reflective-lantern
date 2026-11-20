"""Generate the Veritas-Rag architecture diagram as a PNG."""

from __future__ import annotations

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def draw_box(ax, x, y, w, h, label, color, fontsize=9):
    """Draw a rounded box with a centered label."""
    box = mpatches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.06",
        facecolor=color,
        edgecolor="#333333",
        linewidth=1.1,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="#1a1a1a",
    )


def draw_arrow(ax, x1, y1, x2, y2, label=""):
    """Draw an arrow with an optional midpoint label."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.5},
    )
    if label:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.12,
            label,
            ha="center",
            fontsize=7.5,
            color="#444444",
            style="italic",
        )


def main() -> None:
    """Render architecture.png into the screenshots directory."""
    os.makedirs("screenshots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")
    plt.title(
        "Veritas-Rag — Near-Zero-Hallucination RAG Architecture",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )

    # Ingestion row (stage 1)
    draw_box(ax, 0.3, 6.4, 2.2, 1.1, "PDF / Email /\nHTML / Text", "#cfe8ff")
    draw_box(ax, 3.0, 6.4, 2.2, 1.1, "1. Extract +\nNormalize", "#d8f2d8")
    draw_box(ax, 5.7, 6.4, 2.2, 1.1, "Dedup\n(SHA + MinHash LSH)", "#d8f2d8")
    draw_box(ax, 8.4, 6.4, 2.2, 1.1, "Versioning\n(chains)", "#d8f2d8")
    draw_box(ax, 11.1, 6.4, 2.2, 1.1, "Chunker\n(page-aware)", "#d8f2d8")
    draw_box(ax, 13.7, 6.4, 2.0, 1.1, "Indexes\nBM25 + FAISS", "#ffe9c7")

    # Query row (stages 2-4)
    draw_box(ax, 0.3, 4.2, 2.0, 1.1, "Query", "#cfe8ff")
    draw_box(ax, 2.8, 4.2, 2.0, 1.1, "9. Cache\n(LRU+TTL)", "#fdf3c7")
    draw_box(ax, 5.3, 4.2, 2.2, 1.1, "2. Hybrid\nBM25 + ANN", "#ffe9c7")
    draw_box(ax, 8.0, 4.2, 2.0, 1.1, "RRF Fusion", "#ffe9c7")
    draw_box(ax, 10.5, 4.2, 2.0, 1.1, "3. Rerank\n(deep score)", "#ffd9d9")
    draw_box(ax, 13.0, 4.2, 2.7, 1.1, "4. Trust Score\nfresh+source+consist", "#f3d8f2")

    # Answer row (stages 5-7)
    draw_box(ax, 13.0, 2.0, 2.7, 1.1, "7. Evidence Gate\nthreshold check", "#ffd9d9")
    draw_box(ax, 9.3, 2.0, 2.9, 1.1, "5. Constrained Gen\n(evidence only)", "#d8f2d8")
    draw_box(ax, 5.9, 2.0, 2.6, 1.1, "6. Citations\ndoc + page", "#d8f2d8")
    draw_box(ax, 2.8, 2.0, 2.3, 1.1, "Answer /\nInsufficient evidence", "#cfe8ff")

    # Bottom row (stages 8, 10)
    draw_box(ax, 5.9, 0.2, 3.2, 1.0, "8. Continuous Eval\nrecall/halluc/adversarial", "#e2e2e2")
    draw_box(ax, 9.9, 0.2, 3.2, 1.0, "10. Observability\ntraces + attribution", "#e2e2e2")

    # Ingestion arrows
    draw_arrow(ax, 2.5, 6.95, 3.0, 6.95)
    draw_arrow(ax, 5.2, 6.95, 5.7, 6.95)
    draw_arrow(ax, 7.9, 6.95, 8.4, 6.95)
    draw_arrow(ax, 10.6, 6.95, 11.1, 6.95)
    draw_arrow(ax, 13.3, 6.95, 13.7, 6.95)

    # Query arrows
    draw_arrow(ax, 2.3, 4.75, 2.8, 4.75)
    draw_arrow(ax, 4.8, 4.75, 5.3, 4.75, "miss")
    draw_arrow(ax, 7.5, 4.75, 8.0, 4.75)
    draw_arrow(ax, 10.0, 4.75, 10.5, 4.75)
    draw_arrow(ax, 12.5, 4.75, 13.0, 4.75)
    draw_arrow(ax, 14.7, 6.4, 14.35, 5.3, "search")

    # Answer arrows
    draw_arrow(ax, 14.35, 4.2, 14.35, 3.1)
    draw_arrow(ax, 13.0, 2.55, 12.2, 2.55, "pass")
    draw_arrow(ax, 9.3, 2.55, 8.5, 2.55)
    draw_arrow(ax, 5.9, 2.55, 5.1, 2.55)

    plt.savefig("screenshots/architecture.png", dpi=150, bbox_inches="tight")
    print("Wrote screenshots/architecture.png")


if __name__ == "__main__":
    main()
