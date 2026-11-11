FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_PATH=/app/models/risk_sense_model.pkl
ENV PIPELINE_PATH=/app/models/risk_sense_pipeline.pkl
RUN mkdir -p /app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
