import json
import os
from datetime import datetime

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.logger import setup_logger
from src.pipeline import build_pipeline
from src.utils import load_config, load_features

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

log_path = f"artifacts/logs/train_{timestamp}.log"

logger = setup_logger(name="train", log_file=log_path)

logger.info("Carregando configuração")

cfg = load_config("config.yaml")

RANDOM_SEED = cfg["random_seed"]
TARGET = cfg["target"]
MODEL_NAME = cfg["best_model"]

logger.info(f"Modelo escolhido: {MODEL_NAME}")

DATASET = "data/processed/dataset_academic.csv"
FEATURES_JSON = "artifacts/feature_cols.json"

logger.info("Carregando features")

features = load_features(FEATURES_JSON)

logger.info("Carregando dataset")

df = pd.read_csv(DATASET)

df = df.dropna(subset=[TARGET])

logger.info(f"Dataset possui {df.shape[0]} linhas")

X = df[features]
y = df[TARGET]

logger.info("Dividindo treino e validação")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

logger.info("Construindo pipeline")

pipeline = build_pipeline(features=features, model_name=MODEL_NAME)

logger.info("Treinando modelo")

pipeline.fit(X_train, y_train)

logger.info("Avaliando modelo")

preds = pipeline.predict(X_val)
probs = pipeline.predict_proba(X_val)[:, 1]

metrics = {
    "roc_auc": float(roc_auc_score(y_val, probs)),
    "f1": float(f1_score(y_val, preds)),
    "precision": float(precision_score(y_val, preds)),
    "recall": float(recall_score(y_val, preds)),
}

logger.info(metrics)

os.makedirs("artifacts/models", exist_ok=True)

model_path = "artifacts/models/best_model.joblib"
report_path = "artifacts/models/best_model_metrics.json"

joblib.dump(pipeline, model_path)

with open(report_path, "w") as f:
    json.dump(metrics, f, indent=4)

logger.info(f"Modelo salvo em: {model_path}")
logger.info(f"Relatório salvo em: {report_path}")
