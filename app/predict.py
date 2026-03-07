import pandas as pd

from app.model_loader import get_features, get_model
from app.schemas import PredictionRequest, PredictionResponse
from src.logger import setup_logger

logger = setup_logger("predict")


def predict(request: PredictionRequest):
    logger.info("Prediction requested")

    model = get_model()
    features = get_features()

    data = pd.DataFrame([request.model_dump()])

    data = data[features]

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    logger.info(f"Prediction={prediction} probability={probability}")

    return PredictionResponse(
        prediction=int(prediction),
        probability=float(probability),
    )
