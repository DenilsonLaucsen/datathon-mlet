import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.logger import setup_logger
from src.pipeline import build_pipeline
from src.utils import load_config, load_features


def train(
    config_path: str = "config.yaml",
    dataset_path: str = "data/processed/dataset_academic.csv",
    features_path: str = "artifacts/feature_cols.json",
    output_dir: str = "artifacts/models",
    model_name: str | None = None,
) -> Dict[str, Any]:
    """
    Treina modelo de ML com pipeline configurado.

    Args:
        config_path: Caminho para arquivo de configuração YAML
        dataset_path: Caminho para dataset processado
        features_path: Caminho para arquivo JSON com features
        output_dir: Diretório para salvar modelo e relatório
        model_name: Nome do modelo. Se None, usa valor de config

    Returns:
        Dict com chaves:
            - model: Pipeline treinado
            - metrics: Dicionário com métricas (roc_auc, f1, precision, recall)
            - model_path: Caminho onde modelo foi salvo
            - report_path: Caminho onde relatório foi salvo
    """
    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_path = f"artifacts/logs/train_{timestamp}.log"
    logger = setup_logger(name="train", log_file=log_path)

    # Carregamento de configuração
    logger.info("Carregando configuração")
    cfg = load_config(config_path)

    RANDOM_SEED = cfg["random_seed"]
    TARGET = cfg["target"]
    MODEL_NAME = model_name or cfg["best_model"]

    logger.info(f"Modelo escolhido: {MODEL_NAME}")

    # Carregamento de features
    logger.info("Carregando features")
    features_file = Path(features_path)

    if features_file.exists():
        logger.info("Arquivo de features encontrado")
        features = load_features(features_path)
    else:
        logger.info("Arquivo de features não encontrado. Gerando automaticamente.")

        df_tmp = pd.read_csv(dataset_path)

        features = [col for col in df_tmp.columns if col != TARGET]

        features_file.parent.mkdir(parents=True, exist_ok=True)

        with open(features_file, "w") as f:
            json.dump({"features": features}, f, indent=4)

        logger.info(f"Features salvas em {features_file}")

    # Carregamento de dataset
    logger.info("Carregando dataset")
    df = pd.read_csv(dataset_path)

    df = df.dropna(subset=[TARGET])

    logger.info(f"Dataset possui {df.shape[0]} linhas")

    X = df[features]
    y = df[TARGET]

    # Divisão treino/validação
    logger.info("Dividindo treino e validação")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Construção do pipeline
    logger.info("Construindo pipeline")
    pipeline = build_pipeline(features=features, model_name=MODEL_NAME)

    # Treinamento
    logger.info("Treinando modelo")
    pipeline.fit(X_train, y_train)

    # Avaliação
    logger.info("Avaliando modelo")
    preds = pipeline.predict(X_val)
    probs = pipeline.predict_proba(X_val)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_val, probs)),
        "f1": float(f1_score(y_val, preds)),
        "precision": float(precision_score(y_val, preds)),
        "recall": float(recall_score(y_val, preds)),
    }

    logger.info(f"Métricas: {metrics}")

    # Persistência
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "best_model.joblib")
    report_path = os.path.join(output_dir, "best_model_metrics.json")

    joblib.dump(pipeline, model_path)

    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Modelo salvo em: {model_path}")
    logger.info(f"Relatório salvo em: {report_path}")

    return {
        "model": pipeline,
        "metrics": metrics,
        "model_path": model_path,
        "report_path": report_path,
    }


if __name__ == "__main__":
    train()
