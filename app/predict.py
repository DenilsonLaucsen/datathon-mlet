from pathlib import Path

import pandas as pd

from app.model_loader import get_features, get_model
from app.schemas import PredictionRequest, PredictionResponse
from src.logger import setup_logger

logger = setup_logger("predict")

PREDICTIONS_PATH = Path("data/processed/latest_predictions.csv")


def predict(request: PredictionRequest):
    logger.info("Prediction requested")

    model = get_model()
    features = get_features()

    data = pd.DataFrame([request.model_dump()])

    log_prediction_input(data)

    data = data[features]

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    logger.info(f"Prediction={prediction} probability={probability}")

    return PredictionResponse(
        prediction=int(prediction),
        probability=float(probability),
    )


def log_prediction_input(data: pd.DataFrame) -> None:
    """
    Registra features recebidas pela API para posterior
    monitoramento de data drift.
    """

    try:
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

        if not PREDICTIONS_PATH.exists():
            data.to_csv(PREDICTIONS_PATH, index=False)
        else:
            data.to_csv(PREDICTIONS_PATH, mode="a", header=False, index=False)

    except Exception as e:
        logger.warning(f"Failed to log prediction input: {e}")
