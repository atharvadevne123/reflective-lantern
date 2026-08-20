FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from app.features import generate_synthetic_data, build_feature_pipeline, prepare_X; \
    from app.model import train_model; import joblib; \
    df = generate_synthetic_data(n=1000); fp = build_feature_pipeline(); \
    X = prepare_X(df, fp, fit=True); y = df['delivery_minutes'].values; \
    train_model(X, y); joblib.dump(fp, 'feature_pipeline.joblib')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
