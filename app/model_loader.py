from pathlib import Path

import joblib

from src.logger import setup_logger

logger = setup_logger("model_loader")

MODEL_PATH = Path("artifacts/models/best_model.joblib")

_model = None


def get_model():
    global _model

    if _model is None:
        logger.info("Loading model from disk")
        _model = joblib.load(MODEL_PATH)

    return _model
