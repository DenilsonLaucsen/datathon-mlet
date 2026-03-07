import json
from pathlib import Path

import joblib

from src.logger import setup_logger

logger = setup_logger("model_loader")

MODEL_PATH = Path("artifacts/models/best_model.joblib")
FEATURES_PATH = Path("artifacts/feature_cols.json")

_model = None
_features = None


def get_model():
    global _model

    if _model is None:
        logger.info("Loading model from disk")
        _model = joblib.load(MODEL_PATH)

    return _model


def get_features():
    global _features

    if _features is None:
        logger.info("Loading feature columns")

        with open(FEATURES_PATH) as f:
            payload = json.load(f)

        _features = payload["features"]

    return _features
