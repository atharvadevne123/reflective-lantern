FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY pipelines/ ./pipelines/

ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=postgresql://claimcast:claimcast@db:5432/claimcast
ENV MODEL_PATH=/app/claim_cast_model.joblib
ENV ANOMALY_PATH=/app/claim_cast_anomaly.joblib
ENV METRICS_PATH=/app/claim_cast_metrics.json

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
