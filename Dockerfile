FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DATABASE_URL=postgresql://voltcast:voltcast@db:5432/voltcast
ENV MODEL_PATH=/app/volt_cast_model.joblib
ENV METRICS_PATH=/app/volt_cast_metrics.json

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
