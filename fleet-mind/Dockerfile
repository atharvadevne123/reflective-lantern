FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DATABASE_URL=sqlite:///./fleet_mind.db
ENV MODEL_PATH=/tmp/fleet_mind_model.joblib
ENV METRICS_PATH=/tmp/fleet_mind_metrics.json
ENV LOG_LEVEL=INFO
ENV LOG_JSON=true

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
