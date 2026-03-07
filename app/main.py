import json
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI

from app.predict import predict
from app.schemas import PredictionRequest, PredictionResponse
from monitoring.drift_check import main as run_drift_check

DRIFT_REPORT = Path("artifacts/monitoring/drift_report.json")

app = FastAPI(title="Student Defasagem Predictor", version="1.0.0")


scheduler = BackgroundScheduler()

scheduler.add_job(run_drift_check, "interval", minutes=1)

scheduler.start()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    return predict(request)


@app.get("/monitoring/drift")
def get_drift_report():
    if not DRIFT_REPORT.exists():
        return {"message": "No drift report available"}

    with open(DRIFT_REPORT) as f:
        drift = json.load(f)

    return drift
