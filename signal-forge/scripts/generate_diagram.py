"""Generate ASCII architecture diagram for Signal-Forge."""


def generate_diagram() -> str:
    """Return ASCII architecture diagram as a string."""
    return """
╔══════════════════════════════════════════════════════════════════╗
║                    Signal-Forge Architecture                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Client ──► FastAPI (/predict /health /metrics /drift-check)      ║
║                  │                                                ║
║                  ├──► Feature Pipeline (sklearn)                  ║
║                  │     ├─ VolatilityFeature (21d rolling std)     ║
║                  │     ├─ MomentumFeature (21d pct change)        ║
║                  │     ├─ VolumeRatioFeature (volume/avg)         ║
║                  │     ├─ MarketCorrelationFeature (rolling)       ║
║                  │     └─ BetaFeature (63d rolling beta)          ║
║                  │                                                ║
║                  ├──► Ensemble Model                              ║
║                  │     ├─ XGBoost (200 trees)                     ║
║                  │     ├─ LightGBM (200 trees)                    ║
║                  │     └─ RandomForest (200 trees)                ║
║                  │          └─ 5-fold CV + AUC-ROC                ║
║                  │                                                ║
║                  ├──► FAISS Index (historical regime lookup)      ║
║                  │                                                ║
║                  └──► PostgreSQL (predictions + drift events)     ║
║                                                                   ║
║  Airflow DAG (weekly):                                            ║
║    fetch_data → train_model → drift_check → rebuild_faiss         ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
"""


def save_diagram(path: str = "screenshots/architecture.txt") -> None:
    """Write architecture diagram to file."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(generate_diagram())
    print(f"Diagram saved to {path}")


if __name__ == "__main__":
    print(generate_diagram())
    save_diagram()
