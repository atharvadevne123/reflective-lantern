"""Generate ASCII architecture diagram for Social-Pulse."""

from __future__ import annotations

DIAGRAM = """
╔══════════════════════════════════════════════════════════╗
║              Social-Pulse Architecture                   ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Client                                                  ║
║    │  POST /predict                                      ║
║    ▼                                                     ║
║  FastAPI (app/main.py)                                   ║
║    │  Correlation ID Middleware                          ║
║    │  Pydantic Validation (PostFeatures)                 ║
║    ▼                                                     ║
║  Feature Pipeline (app/features.py)                      ║
║    ├── TextStatTransformer (17 hand-crafted features)    ║
║    │    cyclical hour/dow, follower bucket, NLP stats    ║
║    └── TF-IDF (200 ngram features, unigrams+bigrams)     ║
║    ▼                                                     ║
║  Ensemble Model (app/model.py)                           ║
║    ├── LightGBM (n=300, lr=0.05)                        ║
║    ├── XGBoost  (n=300, lr=0.05)                        ║
║    ├── RandomForest (n=200)                              ║
║    └── LogisticRegression (fallback)                     ║
║    │  soft VotingClassifier (5-fold CV AUC)              ║
║    ▼                                                     ║
║  Monitoring (app/monitoring.py)                          ║
║    ├── Log prediction → PostgreSQL / SQLite              ║
║    └── KS-test drift on 6 features (weekly)             ║
║    ▼                                                     ║
║  Database (app/database.py)                              ║
║    ├── predictions   (inference log)                     ║
║    ├── drift_records (KS-test results)                   ║
║    └── training_runs (model lineage)                     ║
║                                                          ║
║  Airflow DAG (pipelines/retrain_dag.py)                  ║
║    check_drift → retrain_model → notify                  ║
║    schedule: every Sunday 02:00 UTC                      ║
╚══════════════════════════════════════════════════════════╝
"""


def main() -> None:
    """Print the architecture diagram and save it as a text file."""
    import os

    print(DIAGRAM)
    out_path = os.path.join(os.path.dirname(__file__), "..", "screenshots", "architecture.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(DIAGRAM)
    print(f"Diagram saved to {out_path}")


if __name__ == "__main__":
    main()
