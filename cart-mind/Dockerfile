FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user: a container compromise should not land on uid 0.
RUN useradd --create-home --uid 10001 cartmind && chown -R cartmind:cartmind /app
USER cartmind

ENV DATABASE_URL=postgresql://cartmind:secret@db:5432/cartmind
ENV MODEL_PATH=/app/model.joblib
ENV METRICS_PATH=/app/metrics.json
ENV FAISS_INDEX_PATH=/app/faiss_index.idx
ENV ITEM_IDS_PATH=/app/item_ids.json

EXPOSE 8000

# Report unhealthy until the model and FAISS index have finished loading, so an
# orchestrator holds traffic back rather than routing it into a 503.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,json,sys; \
d=json.load(urllib.request.urlopen('http://localhost:8000/api/v1/health',timeout=4)); \
sys.exit(0 if d.get('model_loaded') and d.get('index_loaded') else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
