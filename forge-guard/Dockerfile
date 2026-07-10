FROM python:3.11-slim

LABEL maintainer="Reflective Lantern <devneatharva@gmail.com>"
LABEL description="Forge-Guard manufacturing defect prediction API"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model.joblib \
    METRICS_PATH=/app/metrics.json

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY pipelines/ ./pipelines/
COPY scripts/ ./scripts/

RUN python -c "from app.features import generate_synthetic_data; from app.features import build_feature_pipeline; df=generate_synthetic_data(); X=build_feature_pipeline().fit_transform(df.drop('defect',axis=1)); print('Feature check OK:', X.shape)"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
