import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.logger import setup_logger
from src.pipeline import build_pipeline
from src.utils import load_config, load_features

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

log_path = f"artifacts/logs/train_baseline_{timestamp}.log"

logger = setup_logger(name="train_baseline", log_file=log_path)

cfg = load_config("config.yaml")

RANDOM_SEED = cfg["random_seed"]
N_SPLITS = cfg["n_splits"]
TARGET = cfg["target"]


PREP_CSV = "data/processed/dataset_academic.csv"
FULL_DATASET = "data/processed/dataset_consolidado.csv"
FEATURES_JSON = "artifacts/feature_cols.json"

academic_cols = load_features(FEATURES_JSON)

if os.path.exists(PREP_CSV):
    df = pd.read_csv(PREP_CSV)

else:
    df_full = pd.read_csv(FULL_DATASET)

    if TARGET not in df_full.columns:
        df_full[TARGET] = (df_full["Defasagem"] < 0).astype(int)

    missing = [c for c in academic_cols if c not in df_full.columns]

    if missing:
        raise ValueError(f"Missing academic cols: {missing}")

    df = df_full[academic_cols + [TARGET]].copy()

df = df.dropna(subset=[TARGET])

X = df[academic_cols]
y = df[TARGET]

pipeline = build_pipeline(features=academic_cols, model_name="logreg")

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

scoring = {"roc_auc": "roc_auc", "auprc": "average_precision", "f1": "f1"}

res = cross_validate(
    pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False, n_jobs=-1
)

metrics_mean = {k: float(np.mean(v)) for k, v in res.items() if k.startswith("test_")}


logger.info("CV Results (Desempenho Acadêmico):")
logger.info(metrics_mean)

pipeline.fit(X, y)

os.makedirs("artifacts/models", exist_ok=True)

joblib.dump(pipeline, "artifacts/models/baseline_academico_model.joblib")

with open("artifacts/models/baseline_academico_report.json", "w") as f:
    json.dump(metrics_mean, f, indent=4)

logger.info("Modelo acadêmico salvo com sucesso.")
