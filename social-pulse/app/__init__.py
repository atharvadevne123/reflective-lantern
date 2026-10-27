"""Social-Pulse: viral social media content prediction API.

Exposes a production-ready FastAPI service that predicts whether a social
media post will go viral using an LightGBM + XGBoost + RandomForest ensemble,
17 NLP features, KS-test drift monitoring, and an Airflow retraining DAG.
"""

from __future__ import annotations

__version__: str = "1.0.0"
__author__: str = "Reflective Lantern"
__all__: list[str] = ["__author__", "__version__"]
