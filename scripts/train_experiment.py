import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.logger import setup_logger
from src.pipeline import build_pipeline
from src.utils import load_config, load_features

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

log_path = f"artifacts/logs/train_experiment_{timestamp}.log"

logger = setup_logger(name="train_experiment", log_file=log_path)

if len(sys.argv) < 2:
    logger.error("Uso: python -m scripts.train_experiment <model_name>")
    logger.error("Exemplo: python -m scripts.train_experiment rf")
    sys.exit(1)

MODEL_NAME = sys.argv[1]

logger.info(f"Executando experimento com modelo: {MODEL_NAME}")

logger.info("Carregando configuração")

cfg = load_config("config.yaml")

RANDOM_SEED = cfg["random_seed"]
N_SPLITS = cfg["n_splits"]
TARGET = cfg["target"]

logger.info("Configuração carregada")

PREP_CSV = "data/processed/dataset_academic.csv"
FULL_DATASET = "data/processed/dataset_consolidado.csv"
FEATURES_JSON = "artifacts/feature_cols.json"

logger.info("Carregando features")

features = load_features(FEATURES_JSON)

logger.info(f"{len(features)} features carregadas")

logger.info("Carregando dataset")

if os.path.exists(PREP_CSV):
    df = pd.read_csv(PREP_CSV)
    logger.info(f"Dataset carregado: {PREP_CSV}")

else:
    df_full = pd.read_csv(FULL_DATASET)

    if TARGET not in df_full.columns:
        logger.info("Criando coluna target")
        df_full[TARGET] = (df_full["Defasagem"] < 0).astype(int)

    missing = [c for c in features if c not in df_full.columns]

    if missing:
        logger.error(f"Colunas ausentes: {missing}")
        raise ValueError(f"Missing cols: {missing}")

    df = df_full[features + [TARGET]].copy()
    logger.info(f"Dataset carregado: {FULL_DATASET}")


df = df.dropna(subset=[TARGET])

logger.info(f"Dataset final possui {df.shape[0]} linhas")

X = df[features]
y = df[TARGET]

logger.info("Construindo pipeline")

pipeline = build_pipeline(features=features, model_name=MODEL_NAME)

logger.info("Iniciando validação cruzada")

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

scoring = {"roc_auc": "roc_auc", "auprc": "average_precision", "f1": "f1"}

res = cross_validate(
    pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1
)

metrics_mean = {k: float(np.mean(v)) for k, v in res.items() if k.startswith("test_")}

logger.info("Resultados da validação cruzada:")
logger.info(metrics_mean)

logger.info("Treinando modelo final")

pipeline.fit(X, y)

os.makedirs("artifacts/models", exist_ok=True)

model_path = f"artifacts/models/{MODEL_NAME}_experiment_model.joblib"
report_path = f"artifacts/models/{MODEL_NAME}_experiment_report.json"

joblib.dump(pipeline, model_path)

with open(report_path, "w") as f:
    json.dump(metrics_mean, f, indent=4)

logger.info(f"Modelo salvo em: {model_path}")
logger.info(f"Relatório salvo em: {report_path}")
