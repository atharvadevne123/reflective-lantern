FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DATABASE_URL=postgresql://cartmind:secret@db:5432/cartmind
ENV MODEL_PATH=/app/model.joblib
ENV METRICS_PATH=/app/metrics.json
ENV FAISS_INDEX_PATH=/app/faiss_index.idx
ENV ITEM_IDS_PATH=/app/item_ids.json

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
