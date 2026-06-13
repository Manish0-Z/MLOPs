import hashlib
import json
import os
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import redis
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from paths import load_config, get_path

config = load_config()

redis_client = None
if config.get("redis"):
    try:
        redis_client = redis.Redis(
            host=config["redis"]["host"],
            port=config["redis"]["port"],
            db=config["redis"]["db"],
            password=config["redis"]["password"],
            decode_responses=config["redis"].get("decode_responses", True),
        )
        redis_client.ping()
        print("Redis connected successfully")
    except Exception as e:
        redis_client = None
        print(f"Redis not available (proceeding without cache): {e}")


def _input_cache_key(input_data: dict) -> str:
    raw = json.dumps(input_data, sort_keys=True, default=str)
    return f"prediction:{hashlib.sha256(raw.encode()).hexdigest()}"


def _get_cached_prediction(key: str):
    if redis_client is None:
        return None
    try:
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass
    return None


def _set_cached_prediction(key: str, result: dict):
    if redis_client is None:
        return
    try:
        ttl = config["redis"].get("cache_ttl_seconds", 3600)
        redis_client.setex(key, ttl, json.dumps(result, default=str))
    except Exception:
        pass


app = FastAPI(
    title="Accident Severity Prediction API",
    description="MLOps pipeline - Model serving endpoint for accident severity prediction",
    version="1.0.0",
)

model = None
preprocessors = None
feature_cols = None
label_encoders = None
scaler = None


class PredictionInput(BaseModel):
    Speed_limit: float
    Weather_conditions: Optional[str] = "Unknown"
    Road_type: Optional[float] = 0
    Urban_or_rural_area: Optional[float] = 1
    Vehicle_type: Optional[float] = 1
    Age_of_Vehicle: Optional[float] = 0
    Engine_Capicity: Optional[float] = 0
    Age_of_casualty: Optional[float] = 30
    Casualty_class: Optional[float] = 1
    Age_of_driver: Optional[float] = 40
    Number_of_Vehicles: float
    Number_of_Casualties: float


class PredictionOutput(BaseModel):
    prediction: int
    probability: Optional[float] = None
    severity_label: str


SEVERITY_MAP = {1: "Fatal", 2: "Serious", 3: "Slight"}


def load_model():
    global model, preprocessors, feature_cols, label_encoders, scaler

    mlflow_config = config["mlflow"]
    mlflow.set_tracking_uri(f"sqlite:///{get_path(mlflow_config['tracking_uri'])}")

    client = mlflow.tracking.MlflowClient()
    try:
        alias_info = client.get_model_version_by_alias("AccidentSeverityModel", "champion")
        model_uri = f"models:/AccidentSeverityModel@champion"
        print(f"Loading model from MLflow: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model (version {alias_info.version}) from MLflow Model Registry")
    except Exception:
        print("No champion alias found, loading from run info")
        run_info = joblib.load(get_path("models", "run_info.pkl"))
        model = mlflow.sklearn.load_model(run_info["model_uri"])

    preprocessors = joblib.load(get_path("models", "preprocessors.pkl"))
    label_encoders = preprocessors["label_encoders"]
    scaler = preprocessors["scaler"]
    feature_cols = preprocessors["numeric_cols"] + preprocessors["categorical_cols"]

    print(f"Model loaded. Expected features ({len(feature_cols)}): {feature_cols}")


@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/")
async def root():
    return {
        "service": "Accident Severity Prediction API",
        "status": "running",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
        },
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    input_dict = input_data.model_dump()

    cache_key = _input_cache_key(input_dict)
    cached = _get_cached_prediction(cache_key)
    if cached is not None:
        return PredictionOutput(**cached)

    input_df = pd.DataFrame([input_dict])

    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_cols]

    for col in label_encoders:
        if col in input_df.columns:
            le = label_encoders[col]
            for i, val in enumerate(input_df[col]):
                if val not in le.classes_:
                    input_df.loc[i, col] = le.classes_[0]

    numeric_cols = preprocessors["numeric_cols"]
    for col in numeric_cols:
        if col in input_df.columns:
            try:
                input_df[col] = input_df[col].astype(float)
            except (ValueError, TypeError):
                input_df[col] = 0.0

    for col in label_encoders:
        le = label_encoders[col]
        try:
            input_df[col] = le.transform(input_df[col].astype(str))
        except (ValueError, TypeError):
            input_df[col] = 0

    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    confidence = float(max(probs))

    result = PredictionOutput(
        prediction=int(prediction),
        probability=round(confidence, 4),
        severity_label=SEVERITY_MAP.get(int(prediction), "Unknown"),
    )
    _set_cached_prediction(cache_key, result.model_dump())
    return result


def deploy(start_server=True):
    print("=" * 60)
    print("STAGE 6: MODEL DEPLOYMENT")
    print("=" * 60)

    load_model()

    host = config["deployment"]["host"]
    port = config["deployment"]["port"]
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"Health Check: http://{host}:{port}/health")
    print("Model deployment completed successfully!")

    if start_server:
        print(f"Starting FastAPI server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
    else:
        print("FastAPI server skipped (pipeline mode). Run directly to start server.")
    return True


if __name__ == "__main__":
    deploy(start_server=True)
