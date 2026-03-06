from fastapi import FastAPI

from app.predict import predict
from app.schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Student Defasagem Predictor", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    return predict(request)
