# Realty-Edge

> Real estate property valuation and investment scoring API using XGBoost-LightGBM-RandomForest ensemble with location feature engineering, FAISS comparable search, KS-drift monitoring, and automated retraining pipelines.

![CI](https://github.com/atharvadevne123/Realty-Edge/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

## Overview

Realty-Edge estimates market value for residential properties using a soft-voting ensemble of XGBoost, LightGBM, and RandomForest models. It computes an investment score from rental yield, amenity quality, and neighbourhood risk, searches for comparable properties via FAISS vector similarity, and monitors for data drift using the Kolmogorov-Smirnov test.

## Architecture

![Architecture](screenshots/architecture.png)

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Pydantic v2, slowapi |
| ML | XGBoost, LightGBM, RandomForest (VotingRegressor) |
| Features | sklearn Pipeline (5 stages, 24 features) |
| Search | FAISS IndexFlatL2 |
| Monitoring | KS-test drift detection |
| Database | SQLAlchemy + PostgreSQL (prod) / SQLite (dev) |
| Scheduling | Airflow DAG (weekly) |
| Infra | Docker, docker-compose |
| CI | GitHub Actions + ruff + pytest |

## Quickstart

```bash
cp .env.example .env
docker-compose up --build
```

API available at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

## Local Development

```bash
pip install -r requirements.txt
cp .env.example .env
make run
```

## API Reference

### `POST /api/v1/predict`

```json
{
  "sqft": 1800,
  "bedrooms": 3,
  "bathrooms": 2.0,
  "lot_size": 5000,
  "year_built": 1990,
  "condition_score": 7.5,
  "zipcode": "94102",
  "city": "San Francisco",
  "state": "CA",
  "school_score": 8.0,
  "transit_score": 9.0,
  "walkability_score": 8.5,
  "crime_rate": 0.3,
  "median_neighborhood_price": 1200000,
  "median_price_per_sqft": 800,
  "avg_rental_yield": 0.05,
  "listing_days": 14
}
```

Response includes `predicted_value`, `investment_score`, `confidence_band_low/high`, `model_version`, and `correlation_id`.

### `POST /api/v1/batch-predict`
Accepts `{"properties": [...]}` — up to 100 properties per request.

### `POST /api/v1/comparable-properties`
Returns FAISS nearest-neighbour properties from the prediction index.

### `GET /api/v1/neighborhood-stats/{zipcode}`
Returns cached median prices, school/transit/walk scores, crime rate, rental yield.

### `GET /api/v1/drift-status`
Returns recent KS-test reports and total prediction count.

### `GET /api/v1/health` / `GET /api/v1/metrics`
Health check and model performance metrics.

## Feature Engineering Pipeline

| Stage | Transformer | Features produced |
|---|---|---|
| 1 | PropertyAgeTransformer | `property_age`, `renovation_age` |
| 2 | RatioFeatureTransformer | `beds_per_bath`, `sqft_per_bed`, `price_ratio_neighborhood` |
| 3 | AmenityCompositeTransformer | `amenity_composite`, `risk_score` |
| 4 | InvestmentPotentialTransformer | `investment_potential` |
| 5 | TierEncoderTransformer | `size_tier`, `age_tier` |

## Model Retraining

The Airflow DAG `realty_edge_weekly_retrain` runs every Monday at 02:00 UTC. It fetches up to 5,000 recent prediction logs, retrains the ensemble, rejects the new model if R2 < 0.70, then runs a KS-test drift check.

## Running Tests

```bash
make test
```


## Anomaly Detection

Use `app.anomaly.detect_valuation_anomaly()` to flag properties whose predicted value deviates significantly from the neighbourhood median using Z-score or IQR methods.

## Time-Series Forecasting

`app.time_series` provides SMA, linear trend, and exponential smoothing forecasts for neighbourhood median price series.

## License

MIT
