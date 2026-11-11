# Risk-Sense

[![CI](https://github.com/atharvadevne123/reflective-lantern/actions/workflows/Risk-Sense-CI.yml/badge.svg)](https://github.com/atharvadevne123/reflective-lantern/actions)

Credit risk scoring and loan default prediction API — XGBoost + LightGBM + Random Forest ensemble with KS-test drift monitoring, automated retraining, and PostgreSQL persistence.

## Overview

Risk-Sense predicts the probability of loan default for credit applicants using an ensemble of gradient boosting models. Features include:

- **Ensemble scoring**: XGBoost + LightGBM + Random Forest soft-voting with isotonic calibration
- **5-stage feature pipeline**: log-income transform, interaction features, outlier clipping, standard scaling
- **Real-time drift detection**: KS-test and PSI monitoring against reference distributions
- **Automated retraining**: drift-triggered and scheduled weekly retraining pipelines
- **REST API**: FastAPI with Pydantic v2 validation, versioned endpoints (`/api/v1/`)
- **PostgreSQL persistence**: all predictions and drift logs stored for audit trails
- **5-fold cross-validation**: AUC-ROC reported during training

## Quick Start

```bash
# Clone and set up
cp .env.example .env
docker-compose up --build

# Score an application
curl -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "debt_to_income": 0.35,
    "payment_history_score": 0.85,
    "credit_utilization": 0.30,
    "account_age_months": 48,
    "num_open_accounts": 4,
    "num_derogatory_marks": 0,
    "annual_income": 75000
  }'
```

**Response:**
```json
{
  "applicant_id": "...",
  "risk_score": 0.1234,
  "risk_category": "low",
  "default_probability": 0.1234,
  "recommendation": "Approve — low risk applicant"
}
```
