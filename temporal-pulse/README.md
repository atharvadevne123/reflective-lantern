# Temporal-Pulse

[![CI](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/ci.yml/badge.svg)](https://github.com/atharvadevne123/reflective-lantern/actions)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Multivariate Time-Series Anomaly Detection and Forecasting API** — built with FastAPI, Isolation Forest + Random Forest ensemble, FAISS nearest-neighbour root cause analysis, KS-test drift detection, and a PostgreSQL backend.

## Overview

Temporal-Pulse ingests streaming sensor readings, extracts a rich feature set (rolling statistics, lag features, rate-of-change, cyclical time encoding, and cross-sensor correlations), scores each observation for anomalies, and returns multi-step forecasts with confidence intervals. All anomaly events are persisted and indexed for fast similarity search.

**Key capabilities:**
- Real-time anomaly scoring with configurable threshold
- Multi-step forecasting with per-tree confidence intervals
- FAISS-based root cause explanation (nearest historical anomalies)
- KS-test data drift monitoring across all features
- Automated daily retraining pipeline (Airflow-compatible)
- Full OpenAPI documentation at `/docs`
