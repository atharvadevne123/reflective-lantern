FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY pipelines/ ./pipelines/

ENV DATABASE_URL=postgresql://cyber:guard@db:5432/cyberguard
ENV MODEL_PATH=/app/model.joblib
ENV METRICS_PATH=/app/metrics.json

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
